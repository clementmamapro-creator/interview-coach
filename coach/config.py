"""Configuration centralisée, lue depuis le fichier .env à la racine du projet."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "oui", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    # --- LLM (Claude) ---
    anthropic_api_key: str = ""
    live_model: str = "claude-sonnet-5-5"      # suggestions en direct
    report_model: str = "claude-opus-5-5"      # rapport de fin de session
    max_tokens_live: int = 450
    candidate_name: str = "le candidat"
    language: str = "fr"                       # langue de l'entretien : fr | en | auto

    # --- Transcription ---
    stt_provider: str = "local"                # local (faster-whisper) | groq
    whisper_model: str = "small"               # tiny | base | small | medium | large-v3
    whisper_device: str = "cpu"                # cpu | cuda
    groq_api_key: str = ""
    groq_model: str = "whisper-large-v3-turbo"

    # --- Audio ---
    mic_device: str = ""                       # sous-chaîne du nom ; vide = micro par défaut
    loopback_device: str = ""                  # sous-chaîne du nom ; vide = sortie par défaut
    silence_ms_them: int = 900                 # silence qui clôt une prise de parole de l'interlocuteur
    silence_ms_me: int = 700

    # --- Comportement ---
    auto_suggest_all: bool = False             # True = suggérer même si ce n'est pas une question
    max_context_chars: int = 60000             # au-delà, on passe en recherche RAG (BM25)
    rag_top_k: int = 5

    # --- Chemins / serveur ---
    knowledge_dir: Path = ROOT / "knowledge"
    sessions_dir: Path = ROOT / "sessions"
    host: str = "127.0.0.1"
    port: int = 8765


def load_settings() -> Settings:
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        live_model=os.getenv("LIVE_MODEL", Settings.live_model),
        report_model=os.getenv("REPORT_MODEL", Settings.report_model),
        max_tokens_live=_int("MAX_TOKENS_LIVE", Settings.max_tokens_live),
        candidate_name=os.getenv("CANDIDATE_NAME", Settings.candidate_name),
        language=os.getenv("LANGUAGE", Settings.language),
        stt_provider=os.getenv("STT_PROVIDER", Settings.stt_provider).lower(),
        whisper_model=os.getenv("WHISPER_MODEL", Settings.whisper_model),
        whisper_device=os.getenv("WHISPER_DEVICE", Settings.whisper_device),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        groq_model=os.getenv("GROQ_MODEL", Settings.groq_model),
        mic_device=os.getenv("MIC_DEVICE", ""),
        loopback_device=os.getenv("LOOPBACK_DEVICE", ""),
        silence_ms_them=_int("SILENCE_MS_THEM", Settings.silence_ms_them),
        silence_ms_me=_int("SILENCE_MS_ME", Settings.silence_ms_me),
        auto_suggest_all=_bool("AUTO_SUGGEST_ALL", Settings.auto_suggest_all),
        max_context_chars=_int("MAX_CONTEXT_CHARS", Settings.max_context_chars),
        rag_top_k=_int("RAG_TOP_K", Settings.rag_top_k),
        port=_int("PORT", Settings.port),
    )
