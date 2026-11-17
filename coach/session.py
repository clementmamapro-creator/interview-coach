"""Orchestrateur : audio -> transcription -> détection de question -> Claude -> interface."""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .config import Settings
from .detector import is_echo, should_suggest

log = logging.getLogger(__name__)

Emit = Callable[[dict], Awaitable[None]]


@dataclass
class Turn:
    speaker: str          # "me" | "them"
    text: str
    start: float
    end: float
    simulated: bool = False


@dataclass
class Metrics:
    """Latences mesurées pour objectiver les performances (utile pour itérer et pour le portfolio)."""
    stt_ms: list = field(default_factory=list)          # durée de transcription par énoncé
    first_token_ms: list = field(default_factory=list)  # fin de question -> 1er mot de la suggestion
    total_ms: list = field(default_factory=list)        # fin de question -> suggestion complète

    @staticmethod
    def _avg(v: list) -> Optional[int]:
        return round(sum(v) / len(v)) if v else None

    def summary(self) -> dict:
        return {k: self._avg(getattr(self, k)) for k in ("stt_ms", "first_token_ms", "total_ms")}


class SessionManager:
    def __init__(self, settings: Settings, engine, emit: Emit, transcriber_factory: Optional[Callable] = None):
        self.s = settings
        self.engine = engine
        self.emit = emit
        self.transcriber_factory = transcriber_factory
        self.turns: list[Turn] = []
        self.job_context = ""
        self.listening = False
        self.metrics = Metrics()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._suggest_task: Optional[asyncio.Task] = None
        self._sources: list = []
        self._stt_queue: "queue.Queue" = queue.Queue()
        self._stt_thread: Optional[threading.Thread] = None
        self._transcriber = None
        self._suggestion_id = 0

    # ------------------------------------------------------------------ audio
    async def start_listening(self) -> None:
        if self.listening:
            return
        from .audio import AudioSource, EnergyVAD, get_loopback, get_mic

        self.loop = asyncio.get_running_loop()
        await self.emit({"type": "status", "text": "Chargement de la transcription…"})
        if self._transcriber is None and self.transcriber_factory:
            self._transcriber = await asyncio.to_thread(self.transcriber_factory)

        def on_utt(u):
            self._stt_queue.put(u)

        def on_level(speaker, level):
            if self.loop:
                asyncio.run_coroutine_threadsafe(self.emit({"type": "level", "speaker": speaker, "value": level}), self.loop)

        self._sources = [
            AudioSource("them", lambda: get_loopback(self.s.loopback_device),
                        EnergyVAD(silence_ms=self.s.silence_ms_them), on_utt, on_level),
            AudioSource("me", lambda: get_mic(self.s.mic_device),
                        EnergyVAD(silence_ms=self.s.silence_ms_me), on_utt, on_level),
        ]
        for src in self._sources:
            src.start()
        self.listening = True
        self._stt_thread = threading.Thread(target=self._stt_worker, daemon=True, name="stt")
        self._stt_thread.start()
        await self.emit({"type": "listening", "value": True})
        await self.emit({"type": "status", "text": "À l'écoute (micro + son du PC)"})

    async def stop_listening(self) -> None:
        for src in self._sources:
            src.stop()
        self._sources = []
        self.listening = False
        self._stt_queue.put(None)
        await self.emit({"type": "listening", "value": False})
        await self.emit({"type": "status", "text": "Écoute arrêtée"})

    def _stt_worker(self) -> None:
        while True:
            u = self._stt_queue.get()
            if u is None:
                break
            t0 = time.perf_counter()
            try:
                text = self._transcriber.transcribe(u.audio)
            except Exception as exc:
                log.exception("Erreur de transcription")
                text = ""
                self._threadsafe_emit({"type": "error", "text": f"Transcription : {exc}"})
            self.metrics.stt_ms.append((time.perf_counter() - t0) * 1000)
            if text and self.loop:
                asyncio.run_coroutine_threadsafe(self.add_turn(u.speaker, text, u.start, u.end), self.loop)

    def _threadsafe_emit(self, event: dict) -> None:
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.emit(event), self.loop)

    # -------------------------------------------------------------- dialogue
    async def add_turn(self, speaker: str, text: str, start: float | None = None, end: float | None = None,
                       simulated: bool = False, force_suggest: bool = False) -> Optional[Turn]:
        now = time.time()
        turn = Turn(speaker, text.strip(), start or now, end or now, simulated)
        if not turn.text:
            return None
        if speaker == "me":
            recent_them = [t for t in self.turns[-6:] if t.speaker == "them"]
            if is_echo(turn.text, turn.start, turn.end, recent_them):
                log.info("Écho ignoré : %s", turn.text)
                return None
        self.turns.append(turn)
        await self.emit({"type": "transcript", **asdict(turn)})
        if speaker == "them" and (force_suggest or should_suggest(turn.text, self.s.auto_suggest_all)):
            self.trigger_suggestion(turn.text, question_end=turn.end)
        return turn

    def trigger_suggestion(self, question: str, question_end: float | None = None) -> None:
        if self._suggest_task and not self._suggest_task.done():
            self._suggest_task.cancel()  # une nouvelle question remplace l'ancienne
        self._suggestion_id += 1
        self._suggest_task = asyncio.create_task(
            self._run_suggestion(self._suggestion_id, question, question_end or time.time()))

    async def suggest_now(self) -> None:
        """Déclenchement manuel sur la dernière intervention de l'interlocuteur."""
        last = next((t for t in reversed(self.turns) if t.speaker == "them"), None)
        if last:
            self.trigger_suggestion(last.text)
        else:
            await self.emit({"type": "error", "text": "Aucune question de l'interlocuteur pour l'instant."})

    async def _run_suggestion(self, sid: int, question: str, question_end: float) -> None:
        await self.emit({"type": "suggestion_start", "id": sid, "question": question})
        first = True
        try:
            async for delta in self.engine.stream_suggestion(question, self.turns, self.job_context):
                if first:
                    first = False
                    self.metrics.first_token_ms.append((time.time() - question_end) * 1000)
                await self.emit({"type": "suggestion_delta", "id": sid, "text": delta})
            self.metrics.total_ms.append((time.time() - question_end) * 1000)
            await self.emit({"type": "suggestion_end", "id": sid, "metrics": self.metrics.summary()})
        except asyncio.CancelledError:
            await self.emit({"type": "suggestion_end", "id": sid, "cancelled": True})
            raise
        except Exception as exc:
            log.exception("Erreur Claude")
            await self.emit({"type": "error", "text": f"Claude : {exc}"})

    # ------------------------------------------------------------ simulation
    async def simulate_next(self) -> None:
        await self.emit({"type": "status", "text": "Le recruteur simulé prépare sa question…"})
        try:
            q = await self.engine.simulated_question(self.turns, self.job_context)
        except Exception as exc:
            await self.emit({"type": "error", "text": f"Simulation : {exc}"})
            return
        await self.add_turn("them", q, simulated=True, force_suggest=True)
        await self.emit({"type": "status", "text": "Répondez à voix haute (ou tapez votre réponse)."})

    # ---------------------------------------------------------------- rapport
    async def report(self) -> str:
        if not any(t.speaker == "me" for t in self.turns):
            md = "_Aucune réponse du candidat enregistrée : rien à analyser pour l'instant._"
        else:
            await self.emit({"type": "status", "text": "Rédaction du rapport…"})
            md = await self.engine.session_report(self.turns, self.job_context)
            m = self.metrics.summary()
            md += ("\n\n## Performances techniques de la session\n"
                   f"- Transcription moyenne : {m['stt_ms']} ms\n"
                   f"- Délai moyen avant le 1er mot de suggestion : {m['first_token_ms']} ms\n"
                   f"- Délai moyen de suggestion complète : {m['total_ms']} ms\n")
        path = self.save(md)
        await self.emit({"type": "report", "markdown": md, "path": str(path)})
        await self.emit({"type": "status", "text": f"Rapport enregistré : {path.name}"})
        return md

    async def summary(self) -> str:
        if not self.turns:
            md = "_Aucun échange enregistré pour l'instant._"
        else:
            await self.emit({"type": "status", "text": "Rédaction du compte rendu…"})
            md = await self.engine.session_summary(self.turns, self.job_context)
        path = self.save(md, suffix="_compte-rendu")
        await self.emit({"type": "report", "markdown": md, "path": str(path)})
        await self.emit({"type": "status", "text": "Compte rendu prêt (bouton Copier)"})
        return md

    def save(self, report_md: str = "", suffix: str = "") -> Path:
        self.s.sessions_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        base = self.s.sessions_dir / f"session_{stamp}"
        base.with_suffix(".json").write_text(json.dumps({
            "job_context": self.job_context,
            "turns": [asdict(t) for t in self.turns],
            "metrics": self.metrics.summary(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path = base.parent / f"{base.name}{suffix}.md"
        if report_md:
            md_path.write_text(report_md, encoding="utf-8")
        return md_path

    async def reset(self) -> None:
        if self._suggest_task and not self._suggest_task.done():
            self._suggest_task.cancel()
        self.turns.clear()
        self.metrics = Metrics()
        await self.emit({"type": "reset"})
