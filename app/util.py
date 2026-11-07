"""Small helpers: time formatting and sentence splitting."""

import re
import time
from datetime import datetime


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S (%A)")


def hhmm(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    if dt.date() == datetime.now().date():
        return dt.strftime("%H:%M")
    return dt.strftime("%a %d.%m. %H:%M")


def fmt_dur(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return f"{s} s"
    m = s // 60
    if m < 60:
        return f"{m} min"
    h, m = divmod(m, 60)
    if h < 48:
        return f"{h} h {m} min" if m else f"{h} h"
    return f"{h // 24} days"


def ago(ts: float) -> str:
    return fmt_dur(time.time() - ts) + " ago"


SENTENCE_BREAK = re.compile(r"[.!?…:;]\s+")
MIN_SENTENCE_CHARS = 12  # merge tiny fragments like "Hi!" into the next sentence


def pop_sentence(pending: str) -> tuple[str | None, str]:
    """Split off the first complete sentence of at least MIN_SENTENCE_CHARS."""
    for m in SENTENCE_BREAK.finditer(pending):
        if m.start() + 1 >= MIN_SENTENCE_CHARS:
            return pending[: m.start() + 1], pending[m.end():]
    return None, pending


def split_sentences(text: str) -> list[str]:
    out, rest = [], text
    while True:
        sentence, rest = pop_sentence(rest)
        if sentence is None:
            break
        out.append(sentence)
    if rest.strip():
        out.append(rest)
    return out


EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]+")


def clean_for_speech(text: str) -> str:
    return EMOJI.sub("", re.sub(r"[*_#`>]+", "", text)).strip()


def word_similarity(a: str, b: str) -> float:
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))
