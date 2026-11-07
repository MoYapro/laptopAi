"""Face detection (YuNet) + recognition (ArcFace) and a presence tracker.

Recognition works like a face-login enrollment, minus liveness checks (which need an IR/3D camera):
- ArcFace w600k_r50 (InsightFace), 512-d embeddings from faces aligned to the standard 5-point template
- only good views (big, sharp, roughly frontal) are used to enroll or keep learning a person
- a match must pass the similarity threshold AND beat the second-best person by a margin
- the presence tracker additionally needs several consecutive frames before announcing anyone

YuNet ships with OpenCV; ArcFace runs on onnxruntime. Both on CPU: ~10 ms + ~45 ms per face.
"""

import itertools
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import onnxruntime as ort

import config
from db import DB

log = logging.getLogger("faces")

# ArcFace's canonical landmark positions in a 112x112 crop: eye (image left), eye (image right),
# nose, mouth corner (image left), mouth corner (image right). YuNet emits its 5 points in that order.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    embedding: np.ndarray | None  # normalized 512-d, None if the face is too small to recognize
    good: bool = False  # good enough to enroll / learn from
    quality: dict = field(default_factory=dict)


class FaceEngine:
    def __init__(self):
        d = config.FACE_MODELS_DIR
        self.detector = cv2.FaceDetectorYN.create(str(d / "yunet.onnx"), "", (320, 320), 0.8, 0.3, 50)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.FACE_RECOGNITION_THREADS
        self.session = ort.InferenceSession(str(d / "w600k_r50.onnx"), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def align(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        matrix, _ = cv2.estimateAffinePartial2D(landmarks, ARCFACE_TEMPLATE, method=cv2.LMEDS)
        return cv2.warpAffine(frame, matrix, (112, 112), borderValue=0.0)

    def embed(self, crop: np.ndarray) -> np.ndarray:
        blob = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5
        vec = self.session.run(None, {self.input_name: blob.transpose(2, 0, 1)[None]})[0][0]
        return vec / (np.linalg.norm(vec) + 1e-9)

    def analyze(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        out = []
        for f in faces if faces is not None else []:
            x, y, fw, fh = (int(v) for v in f[:4])
            size = min(fw, fh)
            if size < config.FACE_MIN_SIZE_PX:
                out.append(Detection((x, y, fw, fh), None))
                continue
            lm = f[4:14].reshape(5, 2).astype(np.float32)
            crop = self.align(frame, lm)
            eye_dist = float(np.linalg.norm(lm[1] - lm[0])) + 1e-6
            yaw = abs(float(lm[2][0] - (lm[0][0] + lm[1][0]) / 2)) / eye_dist
            sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
            quality = {"size": size, "score": round(float(f[14]), 2), "yaw": round(yaw, 2), "sharp": round(sharpness)}
            good = (
                size >= config.FACE_ENROLL_MIN_SIZE_PX
                and f[14] >= 0.9
                and yaw <= config.FACE_ENROLL_MAX_YAW
                and sharpness >= config.FACE_ENROLL_MIN_SHARPNESS
            )
            out.append(Detection((x, y, fw, fh), self.embed(crop), good, quality))
        return out


class Gallery:
    """Known faces: several embeddings per person (one model only), matched by best cosine similarity."""

    MAX_PER_PERSON = 40
    LEARN_EVERY_S = 60

    def __init__(self, db: DB):
        self.db = db
        self._vecs: dict[int, list[np.ndarray]] = {}
        self._last_learned: dict[int, float] = {}
        for pid, vec in db.embeddings(config.FACE_MODEL):
            self._vecs.setdefault(pid, []).append(vec)
        log.info("face gallery: %d people, %d embeddings (%s)",
                 len(self._vecs), sum(map(len, self._vecs.values())), config.FACE_MODEL)

    def match(self, emb: np.ndarray) -> tuple[int | None, float]:
        """Best person if they pass the threshold and clearly beat the runner-up; else (None, best_sim)."""
        scores = sorted(
            ((float(np.max(np.stack(vecs) @ emb)), pid) for pid, vecs in self._vecs.items()), reverse=True
        )
        if not scores:
            return None, -1.0
        best_sim, best_pid = scores[0]
        runner_up = scores[1][0] if len(scores) > 1 else -1.0
        if best_sim >= config.FACE_MATCH_THRESHOLD and best_sim - runner_up >= config.FACE_MATCH_MARGIN:
            return best_pid, best_sim
        return None, best_sim

    def add(self, pid: int, emb: np.ndarray) -> None:
        self._vecs.setdefault(pid, []).append(emb)
        self.db.add_embedding(pid, emb, config.FACE_MODEL)

    def count(self, pid: int) -> int:
        return len(self._vecs.get(pid, []))

    def maybe_learn(self, pid: int, det: Detection, sim: float) -> None:
        """Add good, confident-but-not-identical views over time, so recognition adapts to lighting/angles."""
        now = time.time()
        if (
            det.good
            and sim < config.FACE_MATCH_THRESHOLD + 0.3
            and self.count(pid) < self.MAX_PER_PERSON
            and now - self._last_learned.get(pid, 0) > self.LEARN_EVERY_S
        ):
            self._last_learned[pid] = now
            self.add(pid, det.embedding)


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
    embeddings: list[np.ndarray] = field(default_factory=list)  # recent views (matching unknown tracks)
    good_embeddings: list[np.ndarray] = field(default_factory=list)  # recent good views (enrollment)

    def public(self) -> dict:
        return {
            "key": self.key, "name": self.name, "known": self.person_id is not None,
            "since": self.first_seen, "sim": round(self.sim, 2),
        }


class Presence:
    """Turns per-frame detections into stable arrive/leave events."""

    CONFIRM_HITS_KNOWN = 2
    CONFIRM_HITS_UNKNOWN = 4  # strangers need more evidence; a bad angle of a known face shouldn't trigger a greeting
    REUSE_WINDOW_S = 15

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
                self.gallery.maybe_learn(pid, det, sim)
            else:
                track = self._match_unknown(det.embedding, now)
            if track.key in seen:
                continue  # two faces matched one track this frame; keep the first
            seen.add(track.key)
            track.last_seen, track.sim, track.bbox = now, sim, det.bbox
            track.hits += 1
            track.embeddings = (track.embeddings + [det.embedding])[-10:]
            if det.good:
                track.good_embeddings = (track.good_embeddings + [det.embedding])[-10:]
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
        # A turned head often scores below the threshold. If an unknown face vanished seconds ago and a
        # similar-ish one shows up now (and that track isn't already taken this frame), it's the same person.
        if best is not None and best_sim >= 0.2 and best.last_seen < now and now - best.last_seen <= self.REUSE_WINDOW_S:
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
        # Enroll from good views only, like a login enrollment; fall back to the latest views if none were good.
        views = track.good_embeddings or track.embeddings[-3:]
        log.info("enrolling %s from %d views (%d good)", name, len(views), len(track.good_embeddings))
        for emb in views:
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
