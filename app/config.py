"""All settings, read once from the environment (see compose.yaml / .env)."""

import os
from pathlib import Path


def _str(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _float(name: str, default: float) -> float:
    return float(_str(name, str(default)))


def _bool(name: str) -> bool | None:
    return {"true": True, "1": True, "false": False, "0": False}.get(os.environ.get(name, "").strip().lower())


APP_DIR = Path(__file__).parent
DATA_DIR = Path(_str("DATA_DIR", "/data"))
MODELS_DIR = Path(_str("MODELS_DIR", "/models"))
FACE_MODELS_DIR = Path(_str("FACE_MODELS_DIR", "/models/face"))  # downloaded on first start (faces.ensure_face_models)
SNAPSHOT_DIR = DATA_DIR / "snapshots"

# --- models
OLLAMA_URL = _str("OLLAMA_URL", "http://llm:11434")
LLM_MODEL = _str("LLM_MODEL", "qwen3.5:4b")
LLM_THINK = _bool("LLM_THINK")
WHISPER_MODEL = _str("WHISPER_MODEL", "small")
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "").strip() or None
PIPER_VOICE = _str("PIPER_VOICE", "en_US-lessac-medium")
CAMERA_DEVICE = _str("CAMERA_DEVICE", "/dev/video0")

# --- conversation
HISTORY_TURNS = int(_float("HISTORY_TURNS", 10))
HISTORY_MAX_AGE_S = _float("HISTORY_MAX_AGE_S", 6 * 3600)  # older chat is covered by long-term memory

# --- perception
PERCEPTION_INTERVAL_S = _float("PERCEPTION_INTERVAL_S", 1.0)
SCENE_CHANGE_THRESHOLD = _float("SCENE_CHANGE_THRESHOLD", 12.0)  # mean abs pixel diff (0-255) vs last keyframe
SCENE_MIN_INTERVAL_S = _float("SCENE_MIN_INTERVAL_S", 15)
SCENE_MAX_INTERVAL_S = _float("SCENE_MAX_INTERVAL_S", 120)  # re-describe at least this often while people are present
SNAPSHOT_RETENTION_DAYS = _float("SNAPSHOT_RETENTION_DAYS", 7)
FACE_MODEL = "arcface_w600k_r50"  # stored with each embedding; embeddings of other models are ignored
# ArcFace cosine. Measured on this camera: same person median 0.62 (worst poses ~0.36 after enrollment),
# different people <= 0.03. Lookalike relatives can reach 0.3-0.4; the margin rule below guards that case.
FACE_MATCH_THRESHOLD = _float("FACE_MATCH_THRESHOLD", 0.30)
FACE_MATCH_MARGIN = _float("FACE_MATCH_MARGIN", 0.08)  # best person must beat the second best by this much
FACE_MIN_SIZE_PX = _float("FACE_MIN_SIZE_PX", 40)  # smaller faces are detected but not recognized
# Enrollment / learning only from good views (like a login enrollment): big, sharp, roughly frontal.
FACE_ENROLL_MIN_SIZE_PX = _float("FACE_ENROLL_MIN_SIZE_PX", 80)
FACE_ENROLL_MIN_SHARPNESS = _float("FACE_ENROLL_MIN_SHARPNESS", 40)  # Laplacian variance of the aligned crop
FACE_ENROLL_MAX_YAW = _float("FACE_ENROLL_MAX_YAW", 0.25)  # nose offset from the eye midpoint / eye distance
FACE_RECOGNITION_THREADS = int(_float("FACE_RECOGNITION_THREADS", 2))
PERSON_LEFT_AFTER_S = _float("PERSON_LEFT_AFTER_S", 30)

REPLY_TEMPERATURE = _float("REPLY_TEMPERATURE", 0.6)
# Ollama's default context is only 4096 tokens and it silently drops the start of longer prompts
# (that is where the instructions are). Situation + timeline + history + image need more.
CHAT_NUM_CTX = int(_float("CHAT_NUM_CTX", 8192))

# --- memory consolidation (background job with a bigger model, see consolidation.py)
# Preference list: the first model available in Ollama is used (the uncensored 9B is a local LM Studio
# import that only exists on the desktop; the laptop falls back to the registry model).
MEMORY_MODELS = [m.strip() for m in _str("MEMORY_MODEL", "qwen3.5-9b-uncensored,qwen3.5:9b").split(",") if m.strip()]
MEMORY_THINK = _bool("MEMORY_THINK") or False
MEMORY_NUM_CTX = int(_float("MEMORY_NUM_CTX", 16384))
# GPU layers for the memory model: unset = let Ollama decide; 0 = CPU/RAM only (use on the 4 GB laptop,
# so the conversation model keeps the GPU).
MEMORY_NUM_GPU = int(_str("MEMORY_NUM_GPU", "-1"))
CONSOLIDATE_IDLE_S = _float("CONSOLIDATE_IDLE_S", 1800)  # nobody in view and no chat for this long
CONSOLIDATE_INTERVAL_S = _float("CONSOLIDATE_INTERVAL_S", 4 * 3600)
CONSOLIDATE_MAX_ARCHIVE_FRACTION = _float("CONSOLIDATE_MAX_ARCHIVE_FRACTION", 0.3)
BACKUPS_KEEP = int(_float("BACKUPS_KEEP", 10))

# --- initiative
INITIATIVE_REQUIRES_FACE = _bool("INITIATIVE_REQUIRES_FACE") is not False  # never talk to an empty room
INITIATIVE_COOLDOWN_S = _float("INITIATIVE_COOLDOWN_S", 120)
INITIATIVE_MAX_PER_HOUR = int(_float("INITIATIVE_MAX_PER_HOUR", 6))
GREET_AFTER_ABSENCE_S = _float("GREET_AFTER_ABSENCE_S", 1800)  # arrivals after this long bypass the cooldown
CONVERSATION_ACTIVE_S = _float("CONVERSATION_ACTIVE_S", 45)  # no unprompted remarks while someone is chatting
# --- curiosity
REFLECT_INTERVAL_S = _float("REFLECT_INTERVAL_S", 300)  # turn recent scenes into observations + questions
CURIOSITY_IDLE_S = _float("CURIOSITY_IDLE_S", 300)  # quiet this long with someone in view → maybe ask something
QUESTION_REASK_S = _float("QUESTION_REASK_S", 6 * 3600)
QUESTION_MAX_ASKS = int(_float("QUESTION_MAX_ASKS", 2))
MAX_OPEN_QUESTIONS = int(_float("MAX_OPEN_QUESTIONS", 12))

QUIET_DEFAULT =_bool("QUIET_DEFAULT") is not False  # quiet unless explicitly disabled
HEARTBEAT_S = 60
