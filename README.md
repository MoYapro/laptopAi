# laptopai

A local, always-on AI that sees through the webcam, recognizes people, remembers them and speaks up on its own.
Prototyped on the desktop, deployed to an old XPS laptop.

```
browser (localhost:8080)                app container (CPU)                                   llm container (GPU)
  chat / hold-to-talk  ───────────────► faster-whisper (STT) ──► agent ── reply ─────────►  Ollama (qwen3.5:4b)
  timeline, people, camera ◄──────────  perception loop (1/s):                              ▲
  audio playback ◄── WAV per sentence ─   faces (YuNet + ArcFace) ── arrive/leave events    │
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
- **Voice input**: hold the 🎙 button (or Space) to talk, or switch on **listen always**: the page detects speech by
  the microphone level against the room's noise floor and sends each utterance after a 0.9 s pause, without a button.
  It pauses while Housemate thinks or talks (so it doesn't hear itself), drops clicks under 0.35 s and Whisper's usual
  noise hallucinations ("Thank you.", "you"). Thresholds are in `app/static/app.js` (`VAD`).
- **Timeline**: every start (with timestamp and how long it was off), shutdown, unexpected shutdown (detected through a
  heartbeat), scene, arrival/departure, decision and new memory is an event in the database and shown live on the page.
  The model sees the recent timeline in its context.
- **Initiative**: after a startup, an arrival or a scene change, the model decides whether to say something (greet a
  known person and bring up what it remembers, ask a stranger's name, comment on what it sees). Hard limits in code:
  only when a face is in view, not during a conversation, cooldown, max per hour.
  **Quiet mode** (on by default) still runs the decision but only logs what it *would* have said.
- **Curiosity**: every 5 minutes (if new things happened) it reflects on recent scenes, notes observations
  ("A cat lives here.") in short-term memory (the memory review keeps only what recurs hours apart; the rest is
  forgotten after 48 h) and writes down questions ("What's the cat's name?"), shown under
  *Wondering about*. When someone is in view and it's been quiet for 5 minutes, it may ask one. Answers are picked up
  from the conversation; an answer is only accepted if its content actually appears in what the person said.
  Each question is asked at most twice, at least 6 h apart.
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
From the terminal, `project-scripts/send.sh --from "Claude Code" "message"` does exactly that and prints Housemate's reply
(`--camera` attaches the current frame, `--speak` also plays it on open pages, `--timings` prints latency).

Small models copy their own earlier replies, so wrong answers in the saved chat history get repeated. If it keeps
insisting on something false, delete those rows from the `messages` table in `data/laptopai.db`.

## Tuning (environment variables for the app service, defaults in `app/config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `SCENE_CHANGE_THRESHOLD` | 12 | mean pixel difference (0–255) vs the last keyframe that counts as a change |
| `SCENE_MIN_INTERVAL_S` / `SCENE_MAX_INTERVAL_S` | 15 / 120 | min gap between descriptions / re-describe this often while people are present |
| `SNAPSHOT_RETENTION_DAYS` | 7 | snapshot folders older than this are deleted |
| `FACE_MATCH_THRESHOLD` | 0.30 | ArcFace cosine similarity for "same person" (raise if it confuses people) |
| `FACE_MATCH_MARGIN` | 0.08 | the best person must beat the second best by this much |
| `FACE_ENROLL_MIN_SIZE_PX` / `FACE_ENROLL_MAX_YAW` / `FACE_ENROLL_MIN_SHARPNESS` | 80 / 0.25 / 40 | quality bar for enrolling and learning |
| `PERSON_LEFT_AFTER_S` | 30 | unseen this long → "left" |
| `INITIATIVE_COOLDOWN_S` / `INITIATIVE_MAX_PER_HOUR` | 120 / 6 | limits for unprompted remarks |
| `GREET_AFTER_ABSENCE_S` | 1800 | arrivals after this long may bypass the cooldown |
| `QUIET_DEFAULT` | true | quiet mode until switched off on the page |
| `REFLECT_INTERVAL_S` | 300 | how often recent scenes are turned into observations and questions |
| `CURIOSITY_IDLE_S` | 300 | quiet this long with someone in view → it may ask one of its questions |
| `QUESTION_REASK_S` / `QUESTION_MAX_ASKS` | 21600 / 2 | don't re-ask a question sooner / more often than this |

## Internet lookups

Housemate can look up short facts with a `look_up` tool (`app/lookup.py`), only while **internet** is switched on
on the page (off by default):

- DuckDuckGo's Instant Answer API (official, no key), plus a Wikipedia fallback when `LOOKUP_CONTACT` (an email or
  URL for Wikimedia's user-agent policy) is set.
- Results are 1-2 sentences with a `source` field. Housemate uses the source only when asked or when it matters;
  every lookup is in the timeline with its source, so it can answer "where did you get that?" later.
- A relevance check drops articles whose title doesn't match the term (Wikipedia's search always returns *something*),
  and too-long terms are retried shorter. Misses aren't cached; hits are cached for 7 days.
- Lookups never contain names of people from its memory. Results are marked as data, not instructions.
- Not covered: brand-new or niche things, and live data (weather, news, prices).

## Screen

The app runs in a container and can't take screenshots itself, so `project-scripts/screen.sh` is a small helper on
the desktop: it waits until the app asks, then captures each monitor (monitors from `mmsg` on mango or `swaymsg` on
sway, left to right; `-o DP-1` for only one; otherwise one image of everything) and sends them. Needs grim and jq on
Wayland, ImageMagick `import` on X11. Start it with the session, e.g. in mango's `autostart.sh`:
`~/projects/laptopai/project-scripts/screen.sh >/dev/null 2>&1 &`. Without it running, Housemate knows it can't see
the screen.

- **Every minute** (`SCREEN_DESCRIBE_INTERVAL_S`) the app takes a look and describes each monitor that changed, one
  call per monitor ("on the screen: left monitor: … | right monitor: …").
- **When it wants:** the model has a `look_at_screen` tool for a fresh look; a message that refers to the screen
  ("look at this", "what's on my screen", "this error") comes with fresh screenshots right away.
- Frames stay in memory only, each monitor scaled to at most 1280 px (image tokens grow with resolution). The model is
  told to summarize private content and never read out passwords, codes or account numbers.

## Reminders

"Remind me in 20 minutes to take the pasta off" or "… at 18:30": the model calls `set_reminder` (`app/reminders.py`)
with a delay or a clock time; the code computes the due time. When it's due, the model phrases it for the moment and
it is **always spoken**, also to an empty room and in quiet mode. Pending reminders are in the model's situation, so
"cancel the pasta reminder" works (`cancel_reminder`). They are stored in the database and survive restarts; one more
than 6 h overdue (system was off) is logged as missed instead. If the small model answers "Okay, I'll remind you"
without calling the tool, a second, JSON-constrained call picks the request up.

## Memory review (consolidation)

A background job with a bigger model organizes what Housemate knows (`app/consolidation.py`):

- **When:** nobody in view and no chat for 30 min, at most every 4 h, only if something new happened. A background
  run gives way as soon as someone shows up (the running model call is cancelled) and continues in the next quiet
  period. Or now, to the end: `project-scripts/consolidate.sh`, or **Review now** on the memory page (`/memory`).
- **Model:** the first available of `MEMORY_MODEL` (default `qwen3.5-9b-uncensored,qwen3.5:9b`). On the laptop
  `start.sh` sets `MEMORY_NUM_GPU=0`, so it runs on CPU/RAM and the conversation model keeps the 4 GB GPU.
- **Steps:** backup to `data/backups/` → catch up on conversations (facts missed after replies, which topics were
  discussed with whom; developer-tool exchanges are skipped) → turn camera notes seen on separate occasions (6 h+
  apart) into facts → fix facts (merge duplicates, attach to the right
  person, archive trivial/outdated) → sort facts into topics, merge duplicate topics, write profiles → link related topics.
- **Guards in code** (the 9B model doesn't follow these reliably on its own): facts about a moment are skipped, a
  fact starting with a known name belongs to that person, person facts never move to "household", only
  near-identical facts are merged, rewrites can't grow into multi-fact sentences, at most 30% archived per run.
- **Reversible:** nothing is deleted; changed facts are archived with a reason and a pointer to their replacement.
- **Used by the conversation:** for each person in view, their profile, recent facts and the topics talked about with
  them (and related topics) are in the context.

## Face recognition

Works like a face-login enrollment, without liveness detection (that needs an IR/3D camera; a photo of someone
would be recognized as them, which is fine for a home assistant but not for security):

- YuNet finds faces and 5 landmarks; each face is aligned to ArcFace's 112×112 template.
- ArcFace `w600k_r50` (InsightFace) turns it into a 512-d embedding. ~50 ms per face on 2 CPU threads.
- Enrollment and continuous learning only use good views: ≥ 80 px, sharp, roughly frontal, confident detection.
- A match needs similarity ≥ `FACE_MATCH_THRESHOLD` and a margin over the second-best person, and the presence
  tracker needs 2 consecutive frames (4 for strangers) before announcing anyone.
- Measured on the desktop webcam: same person median 0.62, different people ≤ 0.03; after enrolling 5 good views,
  10/10 other views of the same person were recognized.
- Embeddings are tagged with their model; switching models means re-enrolling (name the person once on the page).
- InsightFace's pretrained models are licensed for non-commercial use only.
- The face model files are downloaded on first start into the `models` volume (`/models/face`), checksum-verified,
  with GitHub and Hugging Face as alternative sources. Builds need no model downloads, and a missing model only
  disables face recognition, it doesn't stop the app.

## Privacy

`data/` holds face embeddings, conversation history, memories and camera snapshots. It's git-ignored and stays on
the machine. Delete `data/laptopai.db` to make it forget everything, `data/snapshots/` for the images.

## Measured on the desktop (2026-11-07)

qwen3.5:4b, thinking off: first token ~0.4–0.5 s with the full situation context (~1,100 prompt tokens), first audio
~0.7 s, 70–85 tok/s; Whisper small (CPU, 8 threads) ~1 s per short sentence; face detection ~10 ms per frame.
Expect the laptop to be noticeably slower: measure there.
