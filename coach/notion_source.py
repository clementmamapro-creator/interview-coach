"""Import du dossier candidat directement depuis Notion (aucun fichier local nécessaire).

On lit une ou plusieurs pages racines (ex. « Mon CV décortiqué ») et toutes leurs sous-pages,
puis on les convertit en texte pour la base de connaissances.
Prérequis : une intégration Notion (token) avec laquelle les pages ont été partagées.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import httpx

log = logging.getLogger(__name__)

API = "https://api.notion.com/v1"
VERSION = "2022-06-28"

TEXT_BLOCKS = {
    "paragraph": "", "heading_1": "# ", "heading_2": "## ", "heading_3": "### ",
    "bulleted_list_item": "- ", "numbered_list_item": "1. ", "to_do": "- [ ] ",
    "toggle": "▸ ", "quote": "> ", "callout": "💡 ", "code": "",
}


def normalize_id(page: str) -> str:
    """Accepte une URL Notion ou un identifiant, avec ou sans tirets."""
    m = re.findall(r"[0-9a-fA-F]{32}|[0-9a-fA-F-]{36}", page)
    return m[-1].replace("-", "") if m else page.strip()


def rich_text(items: list) -> str:
    return "".join(i.get("plain_text", "") for i in items or [])


class NotionSource:
    def __init__(self, token: str, client: Optional[httpx.Client] = None, max_depth: int = 3) -> None:
        self.client = client or httpx.Client(timeout=30)
        self.headers = {"Authorization": f"Bearer {token}", "Notion-Version": VERSION}
        self.max_depth = max_depth

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        for _ in range(5):
            r = self.client.get(f"{API}{path}", headers=self.headers, params=params)
            if r.status_code == 429:  # limite de débit Notion (~3 req/s)
                time.sleep(float(r.headers.get("Retry-After", 1)))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("Notion : trop de requêtes, réessayez plus tard")

    def _children(self, block_id: str) -> list[dict]:
        out, cursor = [], None
        while True:
            params = {"page_size": 100, **({"start_cursor": cursor} if cursor else {})}
            data = self._get(f"/blocks/{block_id}/children", params)
            out += data.get("results", [])
            if not data.get("has_more"):
                return out
            cursor = data.get("next_cursor")

    def page_title(self, page_id: str) -> str:
        data = self._get(f"/pages/{page_id}")
        for prop in data.get("properties", {}).values():
            if prop.get("type") == "title":
                return rich_text(prop.get("title")) or "Sans titre"
        return "Sans titre"

    def _render(self, blocks: list[dict], depth: int, subpages: list, indent: str = "") -> list[str]:
        lines = []
        for b in blocks:
            t = b.get("type")
            if t == "child_page":
                subpages.append((b["id"], b["child_page"].get("title", "Sans titre")))
                continue
            if t in TEXT_BLOCKS:
                txt = rich_text(b[t].get("rich_text"))
                if txt:
                    lines.append(f"{indent}{TEXT_BLOCKS[t]}{txt}")
            elif t == "table_row":
                lines.append(indent + " | ".join(rich_text(c) for c in b[t].get("cells", [])))
            if b.get("has_children") and t not in {"child_page", "child_database"} and depth < 8:
                lines += self._render(self._children(b["id"]), depth + 1, subpages, indent + "  ")
        return lines

    def fetch_tree(self, root: str) -> dict[str, str]:
        """Renvoie {titre: texte} pour la page racine et ses sous-pages (récursif)."""
        docs: dict[str, str] = {}
        queue = [(normalize_id(root), None, 0)]
        while queue:
            pid, title, depth = queue.pop(0)
            title = title or self.page_title(pid)
            subpages: list = []
            text = "\n".join(self._render(self._children(pid), 0, subpages)).strip()
            if text:
                docs[f"Notion — {title}"] = text
            if depth < self.max_depth:
                queue += [(sid, stitle, depth + 1) for sid, stitle in subpages]
        return docs


def load_from_notion(token: str, roots: list[str]) -> dict[str, str]:
    src = NotionSource(token)
    docs: dict[str, str] = {}
    for root in roots:
        try:
            docs.update(src.fetch_tree(root))
        except Exception as exc:
            log.error("Import Notion impossible pour %s : %s", root, exc)
            raise
    return docs
