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

from fastapi import FastAPI, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles

import config
from agent import Agent, Hub
from camera import Camera
from consolidation import Consolidator
from db import DB
from faces import FaceEngine, Gallery, Presence, ensure_face_models
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
llm = Ollama(config.OLLAMA_URL, config.LLM_MODEL, think=config.LLM_THINK, num_ctx=config.CHAT_NUM_CTX)
stt = STT(config.WHISPER_MODEL, config.MODELS_DIR / "whisper", config.WHISPER_LANGUAGE)
tts = TTS(config.PIPER_VOICE, config.MODELS_DIR / "piper", config.PIPER_SPEED, config.PIPER_VARIATION)
presence = Presence(Gallery(db), db)
hub = Hub()
agent = Agent(db, llm, tts, presence, hub, base_prompt)
perception = Perception(camera, None, presence, agent)  # face engine set in startup_sequence
consolidator = Consolidator(agent)


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


async def load_face_engine() -> None:
    try:
        await asyncio.to_thread(ensure_face_models, config.FACE_MODELS_DIR)
        perception.engine = await asyncio.to_thread(FaceEngine)
    except Exception as e:
        log.exception("face recognition unavailable")
        await agent.emit("setting", f"face recognition unavailable: {e} (retries on next start)")


async def startup_sequence() -> None:
    await asyncio.gather(warm_up(), load_face_engine())
    perception_task = asyncio.create_task(perception.run())
    await asyncio.sleep(4)  # give face tracking a few frames to settle
    perception.request("the system just started", bypass_cooldown=True)
    await perception_task


@asynccontextmanager
async def lifespan(_: FastAPI):
    agent.on_start()
    camera.start()
    tasks = [
        asyncio.create_task(startup_sequence()),
        asyncio.create_task(agent.heartbeat()),
        asyncio.create_task(agent.reminders.loop()),
        asyncio.create_task(agent.screen.loop()),
        asyncio.create_task(consolidator.loop()),
    ]
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
        "internet": agent.internet,
        "screen": agent.screen.available,
        "present": [t.public() for t in presence.present()],
    }


@app.get("/api/prompt")
async def prompt(q: str = "") -> PlainTextResponse:
    """The conversation system prompt as it would be sent now; q: a message to answer (steers memory selection)."""
    return PlainTextResponse(agent.system_prompt(query=q))


@app.get("/memory")
async def memory_page() -> FileResponse:
    return FileResponse(config.APP_DIR / "static" / "memory.html")


@app.get("/api/memory")
async def memory() -> dict:
    """Everything Housemate remembers, organized: people (profile, facts, topics), topics, household facts."""
    people = []
    for p in db.people():
        people.append({
            "id": p["id"], "name": p["name"], "profile": p["profile"], "last_seen": p["last_seen"],
            "face_samples": presence.gallery.count(p["id"]),
            "facts": db.memories(p["id"], 100),
            "topics": db.topics_for_person(p["id"], 50),
        })
    return {
        "people": people,
        "topics": db.topics(),
        "household": db.memories(None, 200),
        "notes": db.notes(),
        "questions": agent.curiosity.public(),
        "last_review": float(db.get_setting("consolidated_at", "0")) or None,
        "review_running": consolidator.running,
    }


@app.post("/api/consolidate")
async def consolidate(background: bool = False) -> dict:
    """Run the memory review now (takes a while; returns its summary). background=true runs it like the idle job,
    which pauses as soon as someone is around."""
    return await consolidator.run("background" if background else "manual")


@app.get("/api/screen/next")
async def screen_next(wait: float = 60) -> PlainTextResponse:
    """Long poll of the screen helper (project-scripts/screen.sh): "capture" when the app wants a screenshot."""
    return PlainTextResponse("capture" if await agent.screen.next_request(wait) else "idle")


@app.post("/api/screen")
async def screen(request: Request) -> Response:
    """One capture from the screen helper: a file per monitor, named "<monitor>@<x position>.jpg"."""
    shots = []
    for upload in (await request.form()).getlist("shot"):
        name, _, x = upload.filename.rsplit(".", 1)[0].rpartition("@")
        shots.append((name or "screen", int(x) if x.lstrip("-").isdigit() else 0, await upload.read()))
    try:
        await asyncio.to_thread(agent.screen.receive, shots)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return Response(status_code=204)


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
        "internet": agent.internet,
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
            elif kind == "set_internet":
                await agent.set_internet(bool(msg.get("internet")))
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
