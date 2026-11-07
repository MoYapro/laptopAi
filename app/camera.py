"""Background webcam reader that always holds the most recent frame."""

import logging
import threading
import time

import cv2

log = logging.getLogger("camera")


class Camera:
    def __init__(self, device: str, width: int = 640, height: int = 480, fps: int = 10):
        self.device = device
        self.width, self.height, self.fps = width, height, fps
        self._frame = None
        self._frame_time = 0.0
        self._lock = threading.Lock()

    def start(self) -> None:
        threading.Thread(target=self._run, name="camera", daemon=True).start()

    @property
    def available(self) -> bool:
        # A frame older than a few seconds means the camera stopped delivering.
        return self._frame is not None and time.monotonic() - self._frame_time < 3

    def frame(self):
        """Latest frame (BGR numpy array, not copied: treat as read-only) or None."""
        with self._lock:
            frame = self._frame
        return frame if frame is not None and self.available else None

    def jpeg(self, quality: int = 85) -> bytes | None:
        frame = self.frame()
        return encode_jpeg(frame, quality) if frame is not None else None

    def _run(self) -> None:
        while True:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not cap.isOpened():
                log.warning("cannot open %s, retrying in 5s", self.device)
                time.sleep(5)
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            log.info("camera %s opened", self.device)
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.warning("camera read failed, reopening")
                    break
                with self._lock:
                    self._frame = frame
                    self._frame_time = time.monotonic()
            cap.release()
            time.sleep(1)


def encode_jpeg(frame, quality: int = 85) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None
