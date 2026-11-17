"""Transcription (speech-to-text) : faster-whisper en local, ou Groq (API, très rapide)."""
from __future__ import annotations

import io
import logging
import re
import wave
from typing import Protocol

import numpy as np

from .audio import SAMPLE_RATE
from .config import Settings

log = logging.getLogger(__name__)

# Phrases que Whisper « hallucine » sur du silence ou du bruit.
HALLUCINATIONS = {
    "sous-titres réalisés par la communauté d'amara.org",
    "sous-titrage st' 501",
    "merci d'avoir regardé cette vidéo",
    "merci d'avoir regardé",
    "thank you for watching",
    "thanks for watching",
    "merci.",
    "merci",
    "you",
    "...",
}


def clean_transcript(text: str) -> str:
    """Nettoie une transcription ; renvoie '' si c'est du bruit ou une hallucination."""
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t or t.lower().strip(" .!") in {h.strip(" .!") for h in HALLUCINATIONS}:
        return ""
    if not re.search(r"\w", t):
        return ""
    return t


def to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Transcriber(Protocol):
    def transcribe(self, audio: np.ndarray) -> str: ...


class FasterWhisperTranscriber:
    """Transcription 100 % locale (gratuite, privée). 'small' = bon compromis sur CPU."""

    def __init__(self, model_size: str = "small", language: str = "fr", device: str = "cpu") -> None:
        from faster_whisper import WhisperModel

        compute = "int8" if device == "cpu" else "float16"
        log.info("Chargement de Whisper '%s' (%s)…", model_size, device)
        self.model = WhisperModel(model_size, device=device, compute_type=compute)
        self.language = None if language == "auto" else language

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio, language=self.language, beam_size=1,
            vad_filter=False, condition_on_previous_text=False,
        )
        return clean_transcript(" ".join(s.text.strip() for s in segments))


class GroqTranscriber:
    """Whisper large-v3-turbo hébergé chez Groq : ~200-400 ms par énoncé."""

    URL = "https://api.groq.com/openai/v1/audio/transcriptions"

    def __init__(self, api_key: str, model: str, language: str = "fr") -> None:
        import httpx

        if not api_key:
            raise RuntimeError("GROQ_API_KEY manquante dans .env")
        self.client = httpx.Client(timeout=httpx.Timeout(12, connect=5),
                                   headers={"Authorization": f"Bearer {api_key}"})
        self.model = model
        self.language = None if language == "auto" else language

    def transcribe(self, audio: np.ndarray) -> str:
        data = {"model": self.model, "response_format": "json", "temperature": "0"}
        if self.language:
            data["language"] = self.language
        import httpx

        wav = to_wav_bytes(audio)
        for attempt in range(3):  # réseau instable : on retente avant d'abandonner la phrase
            try:
                r = self.client.post(self.URL, data=data, files={"file": ("audio.wav", wav, "audio/wav")})
                r.raise_for_status()
                break
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise
                log.warning("Groq : délai dépassé, nouvelle tentative (%d/2)", attempt + 1)
        return clean_transcript(r.json().get("text", ""))


def build_transcriber(settings: Settings) -> Transcriber:
    if settings.stt_provider == "groq" and not settings.groq_api_key:
        log.warning("GROQ_API_KEY absente : bascule sur la transcription locale (faster-whisper)")
    elif settings.stt_provider == "groq":
        return GroqTranscriber(settings.groq_api_key, settings.groq_model, settings.language)
    return FasterWhisperTranscriber(settings.whisper_model, settings.language, settings.whisper_device)
