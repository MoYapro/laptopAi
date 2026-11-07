# laptopai

A local, always-on AI that sees through the webcam, recognizes people, remembers them and speaks up on its own.
Prototyped on the desktop, deployed to an old XPS laptop.

```
browser (localhost:8080)                app container (CPU)                                   llm container (GPU)
  chat / hold-to-talk  ───────────────► faster-whisper (STT) ──► agent ── reply ─────────►  Ollama (qwen3.5:4b)
  timeline, people, camera ◄──────────  perception loop (1/s):                              ▲
  audio playback ◄── WAV per sentence ─   faces (YuNet + SFace) ── arrive/leave events      │
                                          scene change ── keyframe + description ──────────┤
                                        agent: initiative decision ────────────────────────┤
                                               memory extraction after each reply ─────────┘
                                        SQLite (data/laptopai.db) + snapshots (data/snapshots/)
```

## How it behaves

- **Perception** runs continuously: about once per second it detects and recognizes faces and measures how much the
  image changed. On a real change (or when someone arrives, or every 2 minutes while people are present), it saves a
  timestamped **snapshot** to `data/snapshots/YYYY-MM-DD/HHMMSS.jpg`, and the vision model describes it, **comparing it
  with the previous keyframe** ("what changed"). Unchanged scenes aren't logged.
- **Timeline**: every start (with timestamp and how long it was off), shutdown, unexpected shutdown (detected through a
  heartbeat), scene, arrival/departure, decision and new memory is an event in the database and shown live on the page.
  The model sees the recent timeline in its context.
- **Initiative**: after a startup, an arrival or a scene change, the model decides whether to say something (greet a
  known person and bring up what it remembers, ask a stranger's name, comment on what it sees). Hard limits in code:
  only when a face is in view, not during a conversation, cooldown, max per hour.
  **Quiet mode** (on by default) still runs the decision but only logs what it *would* have said.
- **Memory**: after each reply, a background step extracts durable facts ("Tom is converting his old XPS…") and names.
  Saying "I'm Anna" while exactly one unknown face is in view links the name to that face; or name a face on the page.
  Chat history survives restarts (last 6 h are sent along; older context comes from memory).

## Run

```bash
project-scripts/start.sh     # first run: detects GPU/camera/ids and writes .env, pulls the model, starts everything
project-scripts/stop.sh      # stops containers (writes the shutdown event); data/ and model volumes are kept
# open http://localhost:8080
docker compose logs -f app   # follow perception / decisions / memory in the terminal
```

`start.sh --reconfigure` regenerates `.env` (old one saved as `.env.bak`), `--configure-only` writes it
without starting. Detected values can be overridden: `GPU=cpu LLM_MODEL=gemma3:4b project-scripts/start.sh --reconfigure`.

| Machine | `COMPOSE_FILE` | Notes |
|---|---|---|
| Desktop (RX 9070) | `compose.yaml:compose.amd.yaml` | ROCm Ollama, reuses `~/.ollama`. Don't also run the standalone `ollama` container. |
| XPS laptop | `compose.yaml:compose.nvidia.yaml` | Needs NVIDIA driver + `nvidia-container-toolkit`. |

To use an Ollama that runs elsewhere instead: remove `COMPOSE_PROFILES=llm` and set
`OLLAMA_URL=http://host.docker.internal:11434`.

## Using the page

- **Enter** sends, **Shift+Enter** for a new line. **Hold 🎙** (or **Space** outside text fields) to talk.
- **quiet** (shared, stored on the server), **camera** (attach the current frame to your messages), **speak** (play
  audio in this browser).
- **In view**: recognized people; unknown faces get a name field. The camera preview shows face boxes with the
  match score.
- **Timeline**: newest first; click a snapshot to enlarge it.
- `data/system_prompt.md` overrides the built-in persona (`app/prompts/system.md`); it's re-read every turn.

## Message sources

Every message reaches the model with a header saying how it arrived (`agent.py: SOURCES`), and the page labels it:

| source | label | meaning for the model |
|---|---|---|
| `voice` | 🎙 spoken | someone in the room talked (speech recognition may mishear) |
| `text` | ⌨ typed | someone in the room typed on the screen |
| `developer` + `sender` | 🛠 *sender* | a developer tool (e.g. Claude Code) testing it, not a person in the room; never links a name to a face |
| internal | (timeline only) | its own initiative checks; camera descriptions are context, not messages |

Tools should identify themselves over the WebSocket: `{"type": "turn", "text": "...", "source": "developer", "sender": "Claude Code"}`.

Small models copy their own earlier replies, so wrong answers in the saved chat history get repeated. If it keeps
insisting on something false, delete those rows from the `messages` table in `data/laptopai.db`.

## Tuning (environment variables for the app service, defaults in `app/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `SCENE_CHANGE_THRESHOLD` | 12 | mean pixel difference (0–255) vs the last keyframe that counts as a change |
| `SCENE_MIN_INTERVAL_S` / `SCENE_MAX_INTERVAL_S` | 15 / 120 | min gap between descriptions / re-describe this often while people are present |
| `SNAPSHOT_RETENTION_DAYS` | 7 | snapshot folders older than this are deleted |
| `FACE_MATCH_THRESHOLD` | 0.38 | SFace cosine similarity for "same person" (raise if it confuses people) |
| `PERSON_LEFT_AFTER_S` | 30 | unseen this long → "left" |
| `INITIATIVE_COOLDOWN_S` / `INITIATIVE_MAX_PER_HOUR` | 120 / 6 | limits for unprompted remarks |
| `GREET_AFTER_ABSENCE_S` | 1800 | arrivals after this long may bypass the cooldown |
| `QUIET_DEFAULT` | true | quiet mode until switched off on the page |

## Privacy

`data/` holds face embeddings, conversation history, memories and camera snapshots. It's git-ignored and stays on
the machine. Delete `data/laptopai.db` to make it forget everything, `data/snapshots/` for the images.

## Measured on the desktop (2026-11-07)

qwen3.5:4b, thinking off: first token ~0.4–0.5 s with the full situation context (~1,100 prompt tokens), first audio
~0.7 s, 70–85 tok/s; Whisper small (CPU, 8 threads) ~1 s per short sentence; face detection ~10 ms per frame.
Expect the laptop to be noticeably slower: measure there.
