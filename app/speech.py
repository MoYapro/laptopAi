"""Speech-to-text (faster-whisper) and text-to-speech (Piper), both on CPU.

Models load lazily on first use so the app starts instantly and text-only
sessions never pay for them.
"""

import io
import logging
import os
import threading
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger("speech")


class STT:
    def __init__(self, model_name: str, model_dir: Path, language: str | None):
        self.model_name, self.model_dir, self.language = model_name, model_dir, language
        self._model = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                log.info("loading whisper %s", self.model_name)
                self._model = WhisperModel(
                    self.model_name,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=min(8, os.cpu_count() or 4),
                    download_root=str(self.model_dir),
                )
            return self._model

    def load(self) -> None:
        self._get()

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self._get().transcribe(
            audio_path, beam_size=1, vad_filter=True, language=self.language
        )
        return " ".join(s.text.strip() for s in segments).strip()


class TTS:
    def __init__(self, voice: str, voice_dir: Path, speed: float = 1.0, variation: float = 1.0):
        self.voice_name, self.voice_dir = voice, voice_dir
        self.speed, self.variation = speed, variation
        self._voice = None
        self._config = None
        self._lock = threading.Lock()

    def _get(self):
        with self._lock:
            if self._voice is None:
                from piper import PiperVoice
                from piper.download_voices import download_voice

                model = self.voice_dir / f"{self.voice_name}.onnx"
                if not model.exists():
                    log.info("downloading piper voice %s", self.voice_name)
                    self.voice_dir.mkdir(parents=True, exist_ok=True)
                    download_voice(self.voice_name, self.voice_dir)
                self._voice = PiperVoice.load(model)
                self._config = self._synthesis_config(self._voice)
            return self._voice

    def _synthesis_config(self, voice):
        """Speed and variation relative to the voice's own defaults (they differ per voice)."""
        from piper import SynthesisConfig

        c = voice.config
        return SynthesisConfig(
            length_scale=c.length_scale / self.speed,
            noise_scale=c.noise_scale * self.variation,
            noise_w_scale=c.noise_w_scale * self.variation,
        )

    def load(self) -> None:
        self._get()

    def synthesize_wav(self, text: str) -> bytes:
        voice = self._get()
        chunks = list(voice.synthesize(text, syn_config=self._config))
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(chunks[0].sample_rate if chunks else voice.config.sample_rate)
            for chunk in chunks:
                pcm = np.clip(chunk.audio_float_array, -1.0, 1.0)
                wav.writeframes((pcm * 32767).astype(np.int16).tobytes())
        return buf.getvalue()
