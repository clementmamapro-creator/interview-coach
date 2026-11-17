"""Capture audio sur deux canaux séparés et découpage en prises de parole.

Principe clé pour distinguer les voix : on n'essaie pas de deviner qui parle
dans un flux mélangé. On capture deux flux distincts :
  - le micro            -> "me"   (vous)
  - la sortie système   -> "them" (l'interlocuteur, via le loopback WASAPI)
Chaque flux passe par un VAD (détection d'activité vocale) qui produit des
"Utterance" (une prise de parole complète) envoyées à la transcription.
"""
from __future__ import annotations

import collections
import logging
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000  # 480 échantillons


@dataclass
class Utterance:
    speaker: str          # "me" | "them"
    audio: np.ndarray     # float32 mono 16 kHz
    start: float          # horodatage (time.time())
    end: float


class EnergyVAD:
    """VAD par énergie RMS avec plancher de bruit adaptatif.

    Léger, sans dépendance, suffisant quand chaque canal ne contient qu'une voix.
    """

    def __init__(
        self,
        silence_ms: int = 800,
        min_speech_ms: int = 350,
        max_utterance_s: float = 25.0,
        threshold_factor: float = 3.0,
        min_threshold: float = 0.006,
        preroll_ms: int = 300,
    ) -> None:
        self.silence_frames_needed = max(1, silence_ms // FRAME_MS)
        self.min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self.threshold_factor = threshold_factor
        self.min_threshold = min_threshold
        self.noise_floor = 0.002
        self.preroll: collections.deque = collections.deque(maxlen=max(1, preroll_ms // FRAME_MS))
        self._reset()

    def _reset(self) -> None:
        self.in_speech = False
        self.buffer: list[np.ndarray] = []
        self.silence_frames = 0
        self.speech_frames = 0
        self.start_time = 0.0

    @property
    def threshold(self) -> float:
        return max(self.min_threshold, self.noise_floor * self.threshold_factor)

    def process(self, frame: np.ndarray, t: float) -> Optional[tuple[np.ndarray, float, float]]:
        """Traite une trame ; renvoie (audio, début, fin) quand une prise de parole se termine."""
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
        is_speech = rms > self.threshold

        if not self.in_speech:
            if is_speech:
                self.in_speech = True
                self.buffer = list(self.preroll) + [frame]
                self.speech_frames = 1
                self.silence_frames = 0
                self.start_time = t - len(self.preroll) * FRAME_MS / 1000
            else:
                # adaptation lente du plancher de bruit, uniquement hors parole
                self.noise_floor = 0.97 * self.noise_floor + 0.03 * rms
                self.preroll.append(frame)
            return None

        self.buffer.append(frame)
        if is_speech:
            self.speech_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1

        ended = self.silence_frames >= self.silence_frames_needed
        too_long = len(self.buffer) >= self.max_frames
        if not (ended or too_long):
            return None

        audio = np.concatenate(self.buffer).astype(np.float32)
        start, enough = self.start_time, self.speech_frames >= self.min_speech_frames
        self._reset()
        self.preroll.clear()
        return (audio, start, t) if enough else None


def _com_init() -> None:
    """soundcard (WASAPI) exige une initialisation COM dans chaque thread sous Windows."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        except Exception:  # pragma: no cover
            pass


def _find(devices, name: str):
    name = name.lower()
    for d in devices:
        if name in d.name.lower():
            return d
    raise RuntimeError(f"Périphérique audio introuvable : '{name}'. Lancez `python run.py --devices`.")


def get_mic(name: str = ""):
    import soundcard as sc

    return _find(sc.all_microphones(), name) if name else sc.default_microphone()


def get_loopback(name: str = ""):
    import soundcard as sc

    if name:
        return _find([m for m in sc.all_microphones(include_loopback=True) if m.isloopback], name)
    return sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)


def list_devices() -> str:
    import soundcard as sc

    lines = ["Micros :"]
    lines += [f"  - {m.name}" for m in sc.all_microphones()]
    lines.append("Sorties (loopback) :")
    lines += [f"  - {m.name}" for m in sc.all_microphones(include_loopback=True) if m.isloopback]
    return "\n".join(lines)


class AudioSource(threading.Thread):
    """Thread de capture d'un périphérique, découpé par le VAD."""

    def __init__(self, speaker: str, device_getter: Callable, vad: EnergyVAD,
                 on_utterance: Callable[[Utterance], None], on_level: Optional[Callable[[str, float], None]] = None):
        super().__init__(daemon=True, name=f"audio-{speaker}")
        self.speaker = speaker
        self.device_getter = device_getter
        self.vad = vad
        self.on_utterance = on_utterance
        self.on_level = on_level
        self.stop_event = threading.Event()
        self.error: Optional[str] = None

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        _com_init()
        try:
            device = self.device_getter()
            log.info("Capture %s sur : %s", self.speaker, device.name)
            last_level = 0.0
            with device.recorder(samplerate=SAMPLE_RATE, blocksize=FRAME_LEN) as rec:
                while not self.stop_event.is_set():
                    data = rec.record(numframes=FRAME_LEN)
                    frame = data.mean(axis=1) if data.ndim > 1 else data
                    frame = frame.astype(np.float32)
                    now = time.time()
                    res = self.vad.process(frame, now)
                    if res is not None:
                        audio, start, end = res
                        self.on_utterance(Utterance(self.speaker, audio, start, end))
                    if self.on_level and now - last_level > 0.15:
                        last_level = now
                        self.on_level(self.speaker, float(np.sqrt(np.mean(frame ** 2))))
        except Exception as exc:  # pragma: no cover - dépend du matériel
            self.error = str(exc)
            log.exception("Erreur de capture audio (%s)", self.speaker)
