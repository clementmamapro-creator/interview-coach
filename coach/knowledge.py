"""Base de connaissances du candidat (CV, fiches projets, notes techniques).

Deux stratégies :
  - base petite  -> tout est injecté dans le prompt système (mis en cache côté Claude)
  - base grande  -> recherche BM25 des passages les plus pertinents (RAG)
BM25 est implémenté ici sans dépendance : pas besoin d'API d'embeddings.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

STOPWORDS = set(
    """le la les un une des du de d l et ou en au aux a à est sont ce cet cette ces pour par sur dans
    avec sans que qui quoi dont il elle ils elles on nous vous je tu me te se mon ma mes ton ta tes son sa ses
    votre vos notre nos leur leurs pas ne plus y the a an of to in on for and or is are was were be been with
    as at by it this that you your i my me we our they their do does did how what""".split()
)

SUPPORTED = {".md", ".txt", ".pdf"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9+#]+", normalize(text)) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Chunk:
    source: str
    text: str


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, source: str, size: int = 900) -> list[Chunk]:
    """Découpe par paragraphes, en regroupant jusqu'à ~size caractères."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if current and len(current) + len(p) > size:
            chunks.append(Chunk(source, current))
            current = ""
        current = f"{current}\n\n{p}" if current else p
        while len(current) > size * 1.6:  # paragraphe géant : découpe dure
            chunks.append(Chunk(source, current[:size]))
            current = current[size:]
    if current:
        chunks.append(Chunk(source, current))
    return chunks


class KnowledgeBase:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory
        self.documents: dict[str, str] = {}
        self.chunks: list[Chunk] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._avgdl = 0.0
        if directory:
            self.load(directory)

    # --- chargement ---
    def load(self, directory: Path) -> None:
        self.documents.clear()
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED and not path.name.lower().startswith("readme"):
                try:
                    self.documents[path.name] = read_file(path)
                except Exception as exc:  # fichier illisible : on l'ignore
                    self.documents[path.name] = f"[Erreur de lecture : {exc}]"
        self.index()

    def add_document(self, name: str, text: str) -> None:
        self.documents[name] = text
        self.index()

    def index(self) -> None:
        self.chunks = [c for name, text in self.documents.items() for c in chunk_text(text, name)]
        self._tf = [Counter(tokenize(c.text)) for c in self.chunks]
        self._df = Counter(term for tf in self._tf for term in tf)
        self._avgdl = sum(sum(tf.values()) for tf in self._tf) / max(1, len(self._tf))

    # --- accès ---
    @property
    def total_chars(self) -> int:
        return sum(len(t) for t in self.documents.values())

    def full_text(self) -> str:
        return "\n\n".join(f"### Document : {name}\n{text.strip()}" for name, text in self.documents.items())

    def search(self, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> list[Chunk]:
        terms = tokenize(query)
        if not terms or not self.chunks:
            return []
        n = len(self.chunks)
        scores = []
        for i, tf in enumerate(self._tf):
            dl = sum(tf.values()) or 1
            s = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / self._avgdl))
            if s > 0:
                scores.append((s, i))
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:k]]

    def summary(self) -> dict:
        return {"documents": list(self.documents), "chunks": len(self.chunks), "chars": self.total_chars}
