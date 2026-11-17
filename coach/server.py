"""Serveur local FastAPI + WebSocket qui pilote la session et sert l'interface."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .config import Settings, load_settings
from .knowledge import KnowledgeBase, read_file
from .session import SessionManager

log = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"


class Hub:
    """Diffuse les événements de session à tous les onglets connectés."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def emit(self, event: dict) -> None:
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


class TextIn(BaseModel):
    text: str


class ContextIn(BaseModel):
    job_context: str = ""


def create_app(settings: Optional[Settings] = None, engine=None, transcriber_factory=None) -> FastAPI:
    settings = settings or load_settings()
    kb = KnowledgeBase(settings.knowledge_dir)
    hub = Hub()
    startup_error = None

    if engine is None:
        try:
            from .llm import ClaudeEngine

            engine = ClaudeEngine(settings, kb)
        except Exception as exc:
            startup_error = str(exc)
    if transcriber_factory is None:
        from .stt import build_transcriber

        def transcriber_factory():
            return build_transcriber(settings)

    session = SessionManager(settings, engine, hub.emit, transcriber_factory)
    notion_state = {"status": "désactivé", "documents": 0}

    async def load_notion() -> None:
        split = lambda v: [r.strip() for r in v.split(",") if r.strip()]
        roots, priority = split(settings.notion_pages), split(settings.notion_priority_pages)
        if not (settings.notion_token and (roots or priority)):
            return
        from .notion_source import load_from_notion

        notion_state["status"] = "chargement…"
        await hub.emit({"type": "status", "text": "Import du dossier depuis Notion…"})
        try:
            docs, prio = {}, {}
            if priority:
                prio = await asyncio.to_thread(load_from_notion, settings.notion_token, priority)
                kb.add_documents(prio, pinned=True)
            if roots:
                docs = await asyncio.to_thread(load_from_notion, settings.notion_token, roots)
                kb.add_documents({k: v for k, v in docs.items() if k not in prio})
            docs = {**docs, **prio}
            notion_state.update(status="ok", documents=len(docs), prioritaires=len(prio))
            await hub.emit({"type": "knowledge", **kb.summary()})
            await hub.emit({"type": "status", "text": f"Notion : {len(docs)} page(s) importée(s)"})
        except Exception as exc:
            notion_state["status"] = f"erreur : {exc}"
            await hub.emit({"type": "error", "text": f"Notion : {exc}"})

    @asynccontextmanager
    async def lifespan(_app):
        task = asyncio.create_task(load_notion())
        yield
        task.cancel()

    app = FastAPI(title="Interview Coach", lifespan=lifespan)
    app.state.session, app.state.kb, app.state.hub = session, kb, hub

    def guard():
        if session.engine is None:
            raise RuntimeError(startup_error or "Moteur Claude indisponible")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    async def state():
        return {
            "listening": session.listening,
            "knowledge": kb.summary(),
            "notion": notion_state,
            "error": startup_error,
            "live_model": settings.live_model,
            "stt": settings.stt_provider,
            "turns": len(session.turns),
        }

    @app.post("/api/context")
    async def set_context(body: ContextIn):
        session.job_context = body.job_context
        return {"ok": True}

    @app.post("/api/listen/start")
    async def listen_start():
        try:
            await session.start_listening()
            return {"ok": True}
        except Exception as exc:
            await hub.emit({"type": "error", "text": f"Audio : {exc}"})
            return {"ok": False, "error": str(exc)}

    @app.post("/api/listen/stop")
    async def listen_stop():
        await session.stop_listening()
        return {"ok": True}

    @app.post("/api/ask")
    async def ask(body: TextIn):
        """Saisir une question au clavier (utile pour tester sans audio)."""
        guard()
        await session.add_turn("them", body.text, force_suggest=True)
        return {"ok": True}

    @app.post("/api/answer")
    async def answer(body: TextIn):
        """Saisir sa propre réponse au clavier (mode entraînement écrit)."""
        await session.add_turn("me", body.text)
        return {"ok": True}

    @app.post("/api/suggest")
    async def suggest():
        guard()
        await session.suggest_now()
        return {"ok": True}

    @app.post("/api/simulate/next")
    async def simulate_next():
        guard()
        await session.simulate_next()
        return {"ok": True}

    @app.post("/api/report")
    async def report():
        guard()
        md = await session.report()
        return {"ok": True, "markdown": md}

    @app.post("/api/reset")
    async def reset():
        await session.reset()
        return {"ok": True}

    @app.post("/api/knowledge/reload")
    async def reload_kb():
        kb.load(settings.knowledge_dir)
        await load_notion()
        return kb.summary()

    @app.post("/api/knowledge/upload")
    async def upload(file: UploadFile):
        settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.knowledge_dir / Path(file.filename or "document.txt").name
        dest.write_bytes(await file.read())
        kb.add_document(dest.name, read_file(dest))
        return kb.summary()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        hub.clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            hub.clients.discard(websocket)

    return app
