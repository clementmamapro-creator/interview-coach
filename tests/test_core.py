"""Tests hors matériel audio et sans clé API (moteur Claude simulé)."""
import asyncio
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from coach.audio import FRAME_LEN, EnergyVAD
from coach.config import Settings
from coach.detector import is_echo, is_question, should_suggest
from coach.knowledge import KnowledgeBase, chunk_text
from coach.server import create_app
from coach.session import SessionManager, Turn
from coach.stt import clean_transcript, to_wav_bytes


# ---------------------------------------------------------------- fakes
class FakeEngine:
    def __init__(self):
        self.questions = []

    async def stream_suggestion(self, question, turns, job_context):
        self.questions.append(question)
        for part in ["**À dire** : ", "réponse ", "simulée"]:
            await asyncio.sleep(0)
            yield part

    async def simulated_question(self, turns, job_context):
        return "Quelle est la différence entre un LEFT JOIN et un INNER JOIN ?"

    async def session_report(self, turns, job_context):
        return "## Synthèse\nNote : 7/10"


def make_session(tmp_path):
    events = []

    async def emit(e):
        events.append(e)

    s = Settings(sessions_dir=tmp_path / "sessions", knowledge_dir=tmp_path)
    return SessionManager(s, FakeEngine(), emit), events


# ---------------------------------------------------------------- VAD
def test_vad_detects_one_utterance():
    vad = EnergyVAD(silence_ms=300, min_speech_ms=150)
    rng = np.random.default_rng(0)
    silence = lambda: (rng.standard_normal(FRAME_LEN) * 0.001).astype(np.float32)
    speech = lambda: (np.sin(np.linspace(0, 60, FRAME_LEN)) * 0.3).astype(np.float32)
    out, t = [], 0.0
    for frame_fn, n in [(silence, 40), (speech, 30), (silence, 20)]:
        for _ in range(n):
            t += 0.03
            r = vad.process(frame_fn(), t)
            if r:
                out.append(r)
    assert len(out) == 1
    audio, start, end = out[0]
    assert len(audio) >= 30 * FRAME_LEN and end > start


def test_vad_ignores_short_click():
    vad = EnergyVAD(silence_ms=300, min_speech_ms=300)
    loud = np.full(FRAME_LEN, 0.5, dtype=np.float32)
    quiet = np.zeros(FRAME_LEN, dtype=np.float32)
    results = [vad.process(f, i * 0.03) for i, f in enumerate([quiet] * 10 + [loud] * 2 + [quiet] * 15)]
    assert not any(results)


# ---------------------------------------------------------------- détection
def test_question_detection():
    assert is_question("Pouvez-vous m'expliquer le fonctionnement d'un index B-tree")
    assert is_question("Alors, parlez-moi de votre projet DataSwipe")
    assert is_question("Tell me about a time you failed")
    assert is_question("C'est quoi un RAG ?")
    assert not is_question("Très bien, merci pour cette réponse.")
    assert not should_suggest("Comment ?")  # trop court
    assert should_suggest("Bon, je vois très bien le contexte de la mission", auto_all=True)


def test_echo_dedup():
    them = [Turn("them", "Quelle est la différence entre Spark et Pandas ?", 10.0, 13.0)]
    assert is_echo("quelle est la différence entre spark et pandas", 10.5, 13.2, them)
    assert not is_echo("Spark distribue les calculs sur un cluster", 14.0, 18.0, them)


# ---------------------------------------------------------------- STT utilitaires
def test_clean_transcript_and_wav():
    assert clean_transcript("Sous-titres réalisés par la communauté d'Amara.org") == ""
    assert clean_transcript("  ...  ") == ""
    assert clean_transcript(" Bonjour   Ferry ") == "Bonjour Ferry"
    wav = to_wav_bytes(np.zeros(16000, dtype=np.float32))
    assert wav[:4] == b"RIFF" and len(wav) > 32000


# ---------------------------------------------------------------- RAG
def test_knowledge_search(tmp_path: Path):
    (tmp_path / "cv.md").write_text(
        "# Expériences\n\nAJIG : application Power Apps pour les comptes rendus des centres opérationnels.\n\n"
        "DataSwipe : pipeline de données Airflow, PySpark et dashboards.\n\n"
        "Certification Power BI niveau expert.", encoding="utf-8")
    kb = KnowledgeBase(tmp_path)
    assert kb.summary()["documents"] == ["cv.md"]
    hits = kb.search("pipeline airflow pyspark", k=2)
    assert hits and "DataSwipe" in hits[0].text
    assert len(chunk_text("a" * 5000, "x", size=900)) >= 5


# ---------------------------------------------------------------- session
def test_session_flow(tmp_path):
    async def scenario():
        sess, events = make_session(tmp_path)
        await sess.add_turn("them", "Pouvez-vous expliquer ce qu'est le prompt caching ?")
        await asyncio.sleep(0.05)
        await sess.add_turn("me", "Le cache évite de retraiter le préfixe du prompt.")
        await sess.add_turn("me", "pouvez-vous expliquer ce qu'est le prompt caching")  # écho -> ignoré
        await sess.simulate_next()
        await asyncio.sleep(0.05)
        md = await sess.report()
        return sess, events, md

    sess, events, md = asyncio.run(scenario())
    types = [e["type"] for e in events]
    assert types.count("suggestion_start") == 2
    text = "".join(e["text"] for e in events if e["type"] == "suggestion_delta")
    assert "réponse simulée" in text
    assert [t.speaker for t in sess.turns] == ["them", "me", "them"]
    assert sess.turns[-1].simulated
    assert "Synthèse" in md and "Performances techniques" in md
    assert list((tmp_path / "sessions").glob("*.json")) and list((tmp_path / "sessions").glob("*.md"))
    assert sess.metrics.summary()["first_token_ms"] is not None


def test_new_question_cancels_previous(tmp_path):
    class SlowEngine(FakeEngine):
        async def stream_suggestion(self, question, turns, job_context):
            for _ in range(50):
                await asyncio.sleep(0.01)
                yield "x"

    async def scenario():
        sess, events = make_session(tmp_path)
        sess.engine = SlowEngine()
        await sess.add_turn("them", "Qu'est-ce qu'un DAG dans Airflow ?")
        await asyncio.sleep(0.03)
        await sess.add_turn("them", "Et comment gérez-vous les reprises sur erreur ?")
        await asyncio.sleep(0.8)
        return events

    events = asyncio.run(scenario())
    ends = [e for e in events if e["type"] == "suggestion_end"]
    assert ends[0].get("cancelled") and not ends[-1].get("cancelled")


# ---------------------------------------------------------------- API
def test_api(tmp_path):
    s = Settings(sessions_dir=tmp_path / "s", knowledge_dir=tmp_path)
    app = create_app(s, engine=FakeEngine(), transcriber_factory=lambda: None)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/state").json()["error"] is None
        with client.websocket_connect("/ws") as ws:
            assert client.post("/api/context", json={"job_context": "AI Engineer"}).json()["ok"]
            assert client.post("/api/ask", json={"text": "Qu'est-ce qu'un embedding ?"}).json()["ok"]
            seen = set()
            for _ in range(10):
                seen.add(ws.receive_json()["type"])
                if "suggestion_end" in seen:
                    break
            assert {"transcript", "suggestion_start", "suggestion_end"} <= seen
        r = client.post("/api/knowledge/upload", files={"file": ("notes.md", b"Kafka : streaming distribue", "text/markdown")})
        assert "notes.md" in r.json()["documents"]


# ---------------------------------------------------------------- Notion
def test_notion_import():
    import httpx
    from coach.notion_source import NotionSource, normalize_id

    root, child, toggle = "a" * 32, "b" * 32, "c" * 32
    rt = lambda s: [{"plain_text": s}]
    pages = {
        root: [{"id": "x1", "type": "heading_2", "has_children": False, "heading_2": {"rich_text": rt("Positionnement")}},
               {"id": child, "type": "child_page", "has_children": True, "child_page": {"title": "6. AJIG"}}],
        child: [{"id": "x2", "type": "paragraph", "has_children": False, "paragraph": {"rich_text": rt("12 modules, 15 à 20 min gagnées")}},
                {"id": toggle, "type": "toggle", "has_children": True, "toggle": {"rich_text": rt("Question : la délégation ?")}}],
        toggle: [{"id": "x3", "type": "bulleted_list_item", "has_children": False, "bulleted_list_item": {"rich_text": rt("Limite de 2 000 lignes")}}],
    }
    calls = {"n": 0}

    def handler(req: httpx.Request):
        calls["n"] += 1
        parts = req.url.path.strip("/").split("/")
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        if parts[1] == "pages":
            return httpx.Response(200, json={"properties": {"title": {"type": "title", "title": rt("Mon CV décortiqué")}}})
        return httpx.Response(200, json={"results": pages[parts[2]], "has_more": False})

    src = NotionSource("t", client=httpx.Client(transport=httpx.MockTransport(handler)))
    docs = src.fetch_tree(f"https://app.notion.com/p/{root}?pvs=204")
    assert set(docs) == {"Notion — Mon CV décortiqué", "Notion — 6. AJIG"}
    assert "Limite de 2 000 lignes" in docs["Notion — 6. AJIG"]
    assert normalize_id("https://app.notion.com/p/40550329f195816a8e70d318ae45e855?pvs=204") == "40550329f195816a8e70d318ae45e855"


def test_pinned_documents_in_system():
    from coach.llm import ClaudeEngine

    kb = KnowledgeBase()
    kb.add_documents({"fiche": "DataSwipe pipeline Airflow " * 400})
    kb.add_documents({"Hubvisory": "Pourquoi Hubvisory : conseil produit"}, pinned=True)
    eng = ClaudeEngine.__new__(ClaudeEngine)
    eng.kb, eng.s = kb, Settings(max_context_chars=1000, rag_top_k=3)
    system = eng._system("instructions")
    assert "Pourquoi Hubvisory" in system[1]["text"] and "DataSwipe" not in system[1]["text"]
    assert "DataSwipe" in eng._rag("airflow pipeline")


def test_named_project_gets_full_fiche():
    from coach.llm import ClaudeEngine

    kb = KnowledgeBase()
    kb.add_documents({"Notion — 1. SNCF GPT : assistant IA interne (RAG)": "Fiche SNCF GPT complète. " * 300,
                      "Notion — 6. AJIG : copilote de comptes rendus": "AJIG 12 modules"})
    assert kb.match_documents("Parle-moi de ton projet SNCF-GPT de A à Z") == ["Notion — 1. SNCF GPT : assistant IA interne (RAG)"]
    eng = ClaudeEngine.__new__(ClaudeEngine)
    eng.kb, eng.s = kb, Settings(max_context_chars=1000, rag_top_k=3)
    ctx = eng._rag("Parle-moi de ton projet SNCF GPT en détail")
    assert "<fiche_complete" in ctx and "AJIG" not in ctx
