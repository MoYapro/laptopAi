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
FACE_MODELS_DIR = Path(_str("FACE_MODELS_DIR", "/opt/face-models"))
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
FACE_MATCH_THRESHOLD = _float("FACE_MATCH_THRESHOLD", 0.38)  # SFace cosine; OpenCV suggests 0.363
FACE_MIN_SIZE_PX = _float("FACE_MIN_SIZE_PX", 40)
PERSON_LEFT_AFTER_S = _float("PERSON_LEFT_AFTER_S", 30)

REPLY_TEMPERATURE = _float("REPLY_TEMPERATURE", 0.6)

# --- initiative
INITIATIVE_REQUIRES_FACE = _bool("INITIATIVE_REQUIRES_FACE") is not False  # never talk to an empty room
INITIATIVE_COOLDOWN_S = _float("INITIATIVE_COOLDOWN_S", 120)
INITIATIVE_MAX_PER_HOUR = int(_float("INITIATIVE_MAX_PER_HOUR", 6))
GREET_AFTER_ABSENCE_S = _float("GREET_AFTER_ABSENCE_S", 1800)  # arrivals after this long bypass the cooldown
CONVERSATION_ACTIVE_S = _float("CONVERSATION_ACTIVE_S", 45)  # no unprompted remarks while someone is chatting
QUIET_DEFAULT = _bool("QUIET_DEFAULT") is not False  # quiet unless explicitly disabled
HEARTBEAT_S = 60
