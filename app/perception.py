"""Always-on perception loop.

Every PERCEPTION_INTERVAL_S: detect and recognize faces (cheap, CPU) and measure
how much the scene changed since the last keyframe. On a real change (or people
arriving, or periodically while people are present), save a timestamped
snapshot, have the vision model describe it, comparing against the previous
keyframe, and then let the agent decide whether to speak up.
"""

import asyncio
import logging
import shutil
import time
from datetime import datetime, timedelta

import cv2
import httpx
import numpy as np

import config
from agent import Agent
from camera import Camera, encode_jpeg
from faces import FaceEngine, Presence
from util import word_similarity

log = logging.getLogger("perception")

CAPTION_DUPLICATE_SIMILARITY = 0.75


class Perception:
    def __init__(self, camera: Camera, engine: FaceEngine, presence: Presence, agent: Agent):
        self.camera, self.engine, self.presence, self.agent = camera, engine, presence, agent
        self.reference: np.ndarray | None = None  # downscaled gray keyframe for change detection
        self.last_describe = 0.0
        self.last_change = 0.0
        self.pending_triggers: list[tuple[str, bool]] = []  # (description, bypass_cooldown)
        self.force_describe = False
        self._last_cleanup = 0.0
        config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    def request(self, trigger: str, bypass_cooldown: bool = False) -> None:
        """Ask for a fresh look followed by an initiative decision (e.g. at startup)."""
        self.pending_triggers.append((trigger, bypass_cooldown))
        self.force_describe = True

    async def run(self) -> None:
        log.info("perception loop running every %.1fs", config.PERCEPTION_INTERVAL_S)
        # Don't describe on the very first tick: startup requests a look a few seconds later anyway.
        self.last_describe = time.time()
        while True:
            started = time.monotonic()
            try:
                await self.tick()
                if time.time() - self._last_cleanup > 3600:
                    self._last_cleanup = time.time()
                    await asyncio.to_thread(self.cleanup_snapshots)
            except httpx.HTTPError as e:
                log.warning("perception: model unavailable (%s)", e)
            except Exception:
                log.exception("perception tick failed")
            await asyncio.sleep(max(0.05, config.PERCEPTION_INTERVAL_S - (time.monotonic() - started)))

    async def tick(self) -> None:
        frame = self.camera.frame()
        if frame is None:
            return
        now = time.time()

        detections = await asyncio.to_thread(self.engine.analyze, frame)
        for kind, track in self.presence.update(detections, now):
            await self.agent.on_presence(kind, track)
            if kind == "arrived":
                long_absent = track.prev_seen is None or now - track.prev_seen > config.GREET_AFTER_ABSENCE_S
                who = track.name or "an unknown person"
                self.pending_triggers.append((f"{who} just arrived", track.person_id is None or long_absent))

        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (64, 48)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        change = 255.0 if self.reference is None else float(np.mean(cv2.absdiff(small, self.reference)))
        if change >= config.SCENE_CHANGE_THRESHOLD:
            self.last_change = now

        since = now - self.last_describe
        people = bool(self.presence.present())
        due = (
            self.force_describe
            or (self.pending_triggers and since >= 5)
            or (change >= config.SCENE_CHANGE_THRESHOLD and since >= config.SCENE_MIN_INTERVAL_S)
            or (people and since >= config.SCENE_MAX_INTERVAL_S)
        )
        if not due or self.agent.user_busy:
            return

        self.force_describe = False
        self.last_describe = now
        self.reference = small
        new_scene = await self.describe(frame, now, change)

        triggers, self.pending_triggers = self.pending_triggers, []
        if triggers:
            text = "; ".join(t for t, _ in triggers)
            await self.agent.consider_initiative(text, bypass_cooldown=any(b for _, b in triggers))
        elif new_scene and people:
            await self.agent.consider_initiative("the scene changed")

    async def describe(self, frame, now: float, change: float) -> bool:
        """Save a keyframe, describe it, log it. Returns False if nothing new was seen."""
        jpg = encode_jpeg(frame, 85)
        prev = self.agent.latest_scene
        prev_jpg = self._load_snapshot(prev)

        caption = await self.agent.describe_scene(jpg, prev, prev_jpg)
        if not caption:
            return False
        if "UNCHANGED" in caption or (
            prev and word_similarity(caption, prev["text"]) >= CAPTION_DUPLICATE_SIMILARITY
        ):
            log.info("scene unchanged: %s", caption)
            return False

        rel = datetime.fromtimestamp(now).strftime("%Y-%m-%d/%H%M%S.jpg")
        path = config.SNAPSHOT_DIR / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpg)
        self.agent.latest_scene = await self.agent.emit(
            "scene", caption, {"snapshot": rel, "change": round(change, 1)}, ts=now
        )
        return True

    @staticmethod
    def _load_snapshot(event: dict | None) -> bytes | None:
        rel = (event or {}).get("data", {}) or {}
        rel = rel.get("snapshot")
        if not rel:
            return None
        path = config.SNAPSHOT_DIR / rel
        return path.read_bytes() if path.exists() else None

    @staticmethod
    def cleanup_snapshots() -> None:
        cutoff = (datetime.now() - timedelta(days=config.SNAPSHOT_RETENTION_DAYS)).strftime("%Y-%m-%d")
        for day in config.SNAPSHOT_DIR.iterdir():
            if day.is_dir() and day.name < cutoff:
                shutil.rmtree(day, ignore_errors=True)
                log.info("deleted snapshots of %s", day.name)

    def annotated_jpeg(self) -> bytes | None:
        frame = self.camera.frame()
        if frame is None:
            return None
        frame = frame.copy()
        for track in self.presence.tracks.values():
            if track.bbox is None:
                continue
            x, y, w, h = track.bbox
            color = (90, 200, 120) if track.person_id else (60, 170, 240)
            label = f"{track.name} {track.sim:.2f}" if track.person_id else ("?" if track.announced else "...")
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return encode_jpeg(frame, 70)
