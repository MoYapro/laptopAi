"""Face detection (YuNet) + recognition (SFace) and a presence tracker.

Both models ship with OpenCV's DNN module and run on the CPU in ~10 ms per face.
"""

import itertools
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

import config
from db import DB

log = logging.getLogger("faces")


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    embedding: np.ndarray | None  # normalized 128-d, None if the face is too small to recognize


class FaceEngine:
    def __init__(self):
        d = config.FACE_MODELS_DIR
        self.detector = cv2.FaceDetectorYN.create(str(d / "yunet.onnx"), "", (320, 320), 0.8, 0.3, 50)
        self.recognizer = cv2.FaceRecognizerSF.create(str(d / "sface.onnx"), "")

    def analyze(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        out = []
        for f in faces if faces is not None else []:
            x, y, fw, fh = (int(v) for v in f[:4])
            emb = None
            if min(fw, fh) >= config.FACE_MIN_SIZE_PX:
                feat = self.recognizer.feature(self.recognizer.alignCrop(frame, f)).flatten()
                emb = feat / (np.linalg.norm(feat) + 1e-9)
            out.append(Detection((x, y, fw, fh), emb))
        return out


class Gallery:
    """Known faces: several embeddings per person, matched by best cosine similarity."""

    MAX_PER_PERSON = 40
    LEARN_EVERY_S = 60

    def __init__(self, db: DB):
        self.db = db
        self._vecs: dict[int, list[np.ndarray]] = {}
        self._last_learned: dict[int, float] = {}
        for pid, vec in db.embeddings():
            self._vecs.setdefault(pid, []).append(vec)

    def match(self, emb: np.ndarray) -> tuple[int | None, float]:
        best_pid, best_sim = None, -1.0
        for pid, vecs in self._vecs.items():
            sim = float(np.max(np.stack(vecs) @ emb))
            if sim > best_sim:
                best_pid, best_sim = pid, sim
        if best_sim >= config.FACE_MATCH_THRESHOLD:
            return best_pid, best_sim
        return None, best_sim

    def add(self, pid: int, emb: np.ndarray) -> None:
        self._vecs.setdefault(pid, []).append(emb)
        self.db.add_embedding(pid, emb)

    def maybe_learn(self, pid: int, emb: np.ndarray, sim: float) -> None:
        """Add confident-but-not-identical views over time, so recognition adapts to lighting/angles."""
        now = time.time()
        if (
            sim < config.FACE_MATCH_THRESHOLD + 0.2
            and len(self._vecs.get(pid, [])) < self.MAX_PER_PERSON
            and now - self._last_learned.get(pid, 0) > self.LEARN_EVERY_S
        ):
            self._last_learned[pid] = now
            self.add(pid, emb)


@dataclass
class Track:
    key: str  # "p<id>" for known people, "u<n>" for unknown faces
    person_id: int | None
    name: str | None
    first_seen: float
    last_seen: float
    prev_seen: float | None = None  # when this person was last seen before this visit
    hits: int = 0
    announced: bool = False  # arrival event emitted
    sim: float = 0.0
    bbox: tuple[int, int, int, int] | None = None
    embeddings: list[np.ndarray] = field(default_factory=list)  # recent views (used when naming unknowns)

    def public(self) -> dict:
        return {
            "key": self.key, "name": self.name, "known": self.person_id is not None,
            "since": self.first_seen, "sim": round(self.sim, 2),
        }


class Presence:
    """Turns per-frame detections into stable arrive/leave events."""

    CONFIRM_HITS_KNOWN = 2
    CONFIRM_HITS_UNKNOWN = 4  # strangers need more evidence; a bad angle of a known face shouldn't trigger a greeting

    def __init__(self, gallery: Gallery, db: DB):
        self.gallery, self.db = gallery, db
        self.tracks: dict[str, Track] = {}
        self._unknown_ids = itertools.count(1)

    def present(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.announced]

    def update(self, detections: list[Detection], now: float) -> list[tuple[str, Track]]:
        """Returns events: ("arrived" | "left", track)."""
        events: list[tuple[str, Track]] = []
        seen: set[str] = set()

        for det in detections:
            if det.embedding is None:
                continue
            pid, sim = self.gallery.match(det.embedding)
            if pid is not None:
                key = f"p{pid}"
                track = self.tracks.get(key)
                if track is None:
                    person = self.db.person(pid)
                    track = self.tracks[key] = Track(
                        key, pid, person["name"] if person else "?", now, now,
                        prev_seen=person["last_seen"] if person else None,
                    )
                self.gallery.maybe_learn(pid, det.embedding, sim)
            else:
                track = self._match_unknown(det.embedding, now)
            if track.key in seen:
                continue  # two faces matched one track this frame; keep the first
            seen.add(track.key)
            track.last_seen, track.sim, track.bbox = now, sim, det.bbox
            track.hits += 1
            track.embeddings = (track.embeddings + [det.embedding])[-10:]
            needed = self.CONFIRM_HITS_KNOWN if track.person_id else self.CONFIRM_HITS_UNKNOWN
            if not track.announced and track.hits >= needed:
                track.announced = True
                events.append(("arrived", track))

        for key, track in list(self.tracks.items()):
            if key not in seen:
                track.bbox = None
            if now - track.last_seen > config.PERSON_LEFT_AFTER_S:
                del self.tracks[key]
                if track.announced:
                    events.append(("left", track))
            elif not track.announced and now - track.last_seen > 5:
                del self.tracks[key]  # never confirmed: flicker, drop silently

        for track in self.present():
            if track.person_id:
                self.db.touch_person(track.person_id, now)
        return events

    def _match_unknown(self, emb: np.ndarray, now: float) -> Track:
        best, best_sim = None, -1.0
        for t in self.tracks.values():
            if t.person_id is None and t.embeddings:
                sim = float(np.max(np.stack(t.embeddings) @ emb))
                if sim > best_sim:
                    best, best_sim = t, sim
        if best is not None and best_sim >= config.FACE_MATCH_THRESHOLD:
            return best
        key = f"u{next(self._unknown_ids)}"
        track = self.tracks[key] = Track(key, None, None, now, now)
        return track

    def name_track(self, key: str, name: str) -> Track | None:
        """Bind an unknown track's face to a (new or existing) person."""
        track = self.tracks.get(key)
        if track is None or track.person_id is not None:
            return None
        person = self.db.get_or_create_person(name)
        for emb in track.embeddings:
            self.gallery.add(person["id"], emb)
        del self.tracks[key]
        new_key = f"p{person['id']}"
        if new_key in self.tracks:  # already tracked under that name (e.g. other angle): merge
            existing = self.tracks[new_key]
            existing.first_seen = min(existing.first_seen, track.first_seen)
            return existing
        track.key, track.person_id, track.name = new_key, person["id"], person["name"]
        self.tracks[new_key] = track
        return track

    def unknown_present(self, max_age_s: float = 10) -> list[Track]:
        now = time.time()
        return [t for t in self.present() if t.person_id is None and now - t.last_seen <= max_age_s]
