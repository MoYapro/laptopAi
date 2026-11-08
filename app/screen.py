"""Screen: what is on the user's computer screen.

The app runs in a container and can't take screenshots of the desktop itself. project-scripts/screen.sh is a small
helper on the host (started with the desktop session): it waits on GET /api/screen/next and, when the app asks,
takes a screenshot of each monitor and POSTs them to /api/screen. So the app decides when to look:
- every SCREEN_DESCRIBE_INTERVAL_S (a minute): each monitor that changed is described into the timeline
  (one call per monitor: with several images in one call the small model mixes up which is which)
- when the model calls look_at_screen, or a message refers to the screen ("look at this"): a fresh capture is
  attached to the conversation
Frames are kept in memory only. Without the helper running, the model is told it can't see the screen.
"""

import asyncio
import base64
import logging
import re
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np

import config
from util import ago

if TYPE_CHECKING:
    from agent import Agent

log = logging.getLogger("screen")

MAX_SIDE = 1280  # image tokens grow with resolution; the 4B model gets slow and the context full above this
HELPER_GONE_S = 90  # the helper polls every minute at the latest; no poll for this long = not running
CAPTURE_TIMEOUT_S = 8
REFERS_TO_SCREEN = re.compile(
    r"\b(screens?|monitors?|display|bildschirm|computer|pc|desktop|website|web ?page|browser|tabs?|"
    r"look at (this|that|it)|see (this|that)|this (error|message|code|page|game|window|site|picture|image|video|chart|"
    r"text)|what do you think (of|about) (this|that)|what('s| is) (this|that)|what am i (doing|working on|playing|"
    r"watching|reading)|(left|right) (side|one)|schau (mal )?(hier|das))\b",
    re.I,
)
DESCRIBE_PROMPT = (
    "This is a screenshot of {which} of someone in the home. Describe in one or two short sentences "
    "(max 40 words) what is on it: which program, game or website, and what they are doing. Summarize private "
    "content (messages, emails, documents) instead of quoting it, and never repeat passwords, codes or account "
    "numbers. No preamble."
)
TOOL = {
    "type": "function",
    "function": {
        "name": "look_at_screen",
        "description": (
            "Take a fresh screenshot of the computer screen (all monitors) and look at it. Use it when someone asks "
            "about their screen or what they are doing on the computer, or when you want to see it."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}


class Frame:
    def __init__(self, jpg: bytes, small: np.ndarray, x: int):
        self.jpg, self.small, self.x, self.ts = jpg, small, x, time.time()

    @property
    def b64(self) -> str:
        return base64.b64encode(self.jpg).decode()


def prepare(data: bytes, x: int) -> Frame:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError("could not encode")
    small = cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return Frame(jpg.tobytes(), small, x)


def positions(names: list[str]) -> dict[str, str]:
    """Monitor names (left to right) -> how to call them: 'left monitor', 'right monitor', 'monitor 2 of 3'."""
    if len(names) == 1:
        return {names[0]: "the screen"}
    if len(names) == 2:
        return {names[0]: "the left monitor", names[1]: "the right monitor"}
    return {n: f"monitor {i} of {len(names)} (from the left)" for i, n in enumerate(names, 1)}


class Screen:
    def __init__(self, agent: "Agent"):
        self.agent = agent
        self.frames: list[tuple[str, Frame]] = []  # the latest capture, left to right
        self.helper_seen = 0.0
        self._want = asyncio.Event()
        self._got = asyncio.Event()
        self._described: dict[str, np.ndarray] = {}  # thumbnails at the last description, per monitor
        self._captions: dict[str, str] = {}
        self.latest: dict | None = agent.db.last_event(["screen"])

    # ------------------------------------------------------- the host helper

    @property
    def available(self) -> bool:
        return time.time() - self.helper_seen < HELPER_GONE_S

    async def next_request(self, wait: float) -> bool:
        """Long poll of the helper: True when the app wants a screenshot."""
        self.helper_seen = time.time()
        try:
            await asyncio.wait_for(self._want.wait(), min(wait, 60))
        except asyncio.TimeoutError:
            return False
        finally:
            self.helper_seen = time.time()
        self._want.clear()
        return True

    def receive(self, shots: list[tuple[str, int, bytes]]) -> None:
        """One capture: (monitor name, x position, image) for each monitor."""
        frames = [(name[:40], prepare(data, x)) for name, x, data in shots]
        if not frames:
            raise ValueError("no images")
        self.frames = sorted(frames, key=lambda nf: nf[1].x)
        self._got.set()

    async def capture(self) -> list[tuple[str, Frame]]:
        """Ask the helper for a fresh screenshot of every monitor; [] if it isn't running or doesn't answer."""
        if not self.available:
            return []
        self._got = got = asyncio.Event()
        self._want.set()
        try:
            await asyncio.wait_for(got.wait(), CAPTURE_TIMEOUT_S)
        except asyncio.TimeoutError:
            log.warning("the screen helper didn't send a screenshot")
            return []
        return self.frames

    # ------------------------------------------------- for the conversation

    async def images_for(self, text: str) -> list[str]:
        """Fresh screenshots (base64, left to right) when a message refers to the screen."""
        if not REFERS_TO_SCREEN.search(text):
            return []
        return [f.b64 for _, f in await self.capture()]

    async def look(self) -> dict:
        """The look_at_screen tool. The images go into the conversation as a separate message ("_images")."""
        frames = await self.capture()
        if not frames:
            return {"ok": False, "reason": "you can't see the screen right now (the screen helper isn't running)"}
        where = positions([n for n, _ in frames])
        log.info("look_at_screen: %d monitors", len(frames))
        return {"ok": True, "attached": [where[n] for n, _ in frames], "_images": [f.b64 for _, f in frames]}

    def context_lines(self) -> list[str]:
        if not self.available:
            return ["Screen: you can't see the computer screen right now (its screen helper isn't running)."]
        line = "Screen: you can see the computer screen; call look_at_screen to take a fresh look."
        if self.latest:
            line += f" Latest screen description ({ago(self.latest['ts'])}): {self.latest['text']}"
        return [line]

    # ----------------------------------------------------- background view

    async def loop(self) -> None:
        while True:
            await asyncio.sleep(config.SCREEN_DESCRIBE_INTERVAL_S)
            try:
                if self.available and not self.agent.user_busy:
                    await self.describe_changes()
            except Exception:
                log.exception("screen description failed")

    async def describe_changes(self) -> None:
        frames = await self.capture()
        changed = [
            (n, f) for n, f in frames
            if n not in self._described
            or float(np.mean(cv2.absdiff(f.small, self._described[n]))) >= config.SCREEN_CHANGE_THRESHOLD
        ]
        if not changed:
            return
        where = positions([n for n, _ in frames])
        for name, frame in changed:
            messages = [
                {"role": "system", "content": "You write the screen log of a home AI. Be factual and concise."},
                {"role": "user", "content": DESCRIBE_PROMPT.format(
                    which="the computer screen" if len(frames) == 1 else f"{where[name]} of a computer"),
                 "images": [frame.b64]},
            ]
            async with self.agent.lock:
                caption = await self.agent.llm.complete(messages, max_tokens=120, temperature=0.2,
                                                        model=await self.agent.llm.vision_model())
            self._captions[name] = re.sub(r"\s+", " ", re.sub(r"[*#_`]+", "", caption)).strip()
            self._described[name] = frame.small
        parts = [f"{where[n].removeprefix('the ')}: {self._captions[n]}" if len(frames) > 1 else self._captions[n]
                 for n, _ in frames if self._captions.get(n)]
        self.latest = await self.agent.emit("screen", "on the screen: " + " | ".join(parts))
