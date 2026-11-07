"""laptopai web app: wiring of camera, perception, agent and the browser UI.

Audio I/O goes through the browser (mic recording, playback), so the container
needs no access to host audio; the kiosk browser on the laptop plays through
its speakers.
"""

import asyncio
import logging
import tempfile
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

import config
from agent import Agent, Hub
from camera import Camera
from db import DB
from faces import FaceEngine, Gallery, Presence
from llm import Ollama
from perception import Perception
from speech import STT, TTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("app")


def base_prompt() -> str:
    # /data/system_prompt.md overrides the built-in prompt; re-read every turn so edits apply live.
    override = config.DATA_DIR / "system_prompt.md"
    return (override if override.exists() else config.APP_DIR / "prompts" / "system.md").read_text()


db = DB(config.DATA_DIR / "laptopai.db")
camera = Camera(config.CAMERA_DEVICE)
llm = Ollama(config.OLLAMA_URL, config.LLM_MODEL, think=config.LLM_THINK)
stt = STT(config.WHISPER_MODEL, config.MODELS_DIR / "whisper", config.WHISPER_LANGUAGE)
tts = TTS(config.PIPER_VOICE, config.MODELS_DIR / "piper")
presence = Presence(Gallery(db), db)
hub = Hub()
agent = Agent(db, llm, tts, presence, hub, base_prompt)
perception = Perception(camera, FaceEngine(), presence, agent)


async def warm_up() -> None:
    """Load every model before perception starts, so the first real turn isn't slow."""
    for attempt in range(60):
        if await llm.ping():
            break
        if attempt == 0:
            log.warning("Ollama not reachable at %s yet, waiting", llm.url)
        await asyncio.sleep(2)
    for name, load in [("llm", llm.load), ("tts", tts.load), ("stt", stt.load)]:
        try:
            t0 = time.perf_counter()
            await (load() if name == "llm" else asyncio.to_thread(load))
            log.info("warm-up %s done in %d ms", name, (time.perf_counter() - t0) * 1000)
        except Exception:
            log.exception("warm-up %s failed", name)


async def startup_sequence() -> None:
    await warm_up()
    perception_task = asyncio.create_task(perception.run())
    await asyncio.sleep(4)  # give face tracking a few frames to settle
    perception.request("the system just started", bypass_cooldown=True)
    await perception_task


@asynccontextmanager
async def lifespan(_: FastAPI):
    agent.on_start()
    camera.start()
    tasks = [asyncio.create_task(startup_sequence()), asyncio.create_task(agent.heartbeat())]
    yield
    for task in tasks:
        task.cancel()
    await agent.on_stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.APP_DIR / "static"), name="static")
config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=config.SNAPSHOT_DIR), name="snapshots")


# --------------------------------------------------------------------------- HTTP


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(config.APP_DIR / "static" / "index.html")


@app.get("/camera.jpg")
async def camera_jpg() -> Response:
    jpg = await asyncio.to_thread(perception.annotated_jpeg)
    if jpg is None:
        return Response(status_code=503)
    return Response(jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/status")
async def status() -> dict:
    return {
        "model": llm.model,
        "ollama": await llm.ping(),
        "camera": camera.available,
        "whisper": stt.model_name,
        "voice": tts.voice_name,
        "quiet": agent.quiet,
        "present": [t.public() for t in presence.present()],
    }


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile) -> JSONResponse:
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".webm") as f:
        f.write(await audio.read())
        f.flush()
        text = await asyncio.to_thread(stt.transcribe, f.name)
    return JSONResponse({"text": text, "ms": round((time.perf_counter() - t0) * 1000)})


# ---------------------------------------------------------------------- WebSocket


class Client:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self._lock = asyncio.Lock()

    async def send(self, msg: dict) -> None:
        async with self._lock:
            await self.ws.send_json(msg)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    client = Client(ws)
    await client.send({
        "type": "hello",
        "messages": db.recent_messages(40),
        "events": db.recent_events(80),
        "people": [t.public() for t in presence.present()],
        "quiet": agent.quiet,
        "questions": agent.curiosity.public(),
        "model": llm.model,
    })
    hub.clients.add(client)
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "turn" and (text := str(msg.get("text", "")).strip()):
                if agent.user_busy:
                    await client.send({"type": "error", "text": "still busy with the previous turn"})
                    continue
                image = camera.jpeg() if msg.get("camera") else None
                if msg.get("camera") and image is None:
                    await client.send({"type": "notice", "text": "camera unavailable, sending text only"})
                asyncio.create_task(_run_turn(text, msg, image))
            elif kind == "set_quiet":
                await agent.set_quiet(bool(msg.get("quiet")))
            elif kind == "name_face" and (name := str(msg.get("name", "")).strip()):
                await agent.learn_name(name, track_key=str(msg.get("key", "")))
    except WebSocketDisconnect:
        pass
    finally:
        hub.clients.discard(client)


async def _run_turn(text: str, msg: dict, image: bytes | None) -> None:
    source = msg.get("source") if msg.get("source") in ("text", "voice", "developer") else "text"
    sender = str(msg.get("sender") or "").strip()[:80] or None
    if source == "developer" and not sender:
        sender = "an unnamed developer tool"
    try:
        await agent.reply(
            text,
            source=source,
            image_jpg=image,
            speak=bool(msg.get("speak")),
            stt_ms=msg.get("stt_ms"),
            sender=sender if source == "developer" else None,
        )
    except Exception as e:  # show pipeline errors in the UI
        log.exception("turn failed")
        await hub.broadcast({"type": "error", "text": str(e)})
