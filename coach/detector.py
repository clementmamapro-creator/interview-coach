"""Détection des questions et dédoublonnage de l'écho micro."""
from __future__ import annotations

import difflib
import re

from .knowledge import normalize

QUESTION_STARTERS = (
    # français
    "qu", "quel", "quelle", "quels", "quelles", "comment", "pourquoi", "combien", "ou ", "quand",
    "est-ce", "pouvez", "pourriez", "peux-tu", "pourrais", "avez-vous", "as-tu", "etes-vous",
    "parlez", "parle-moi", "racontez", "raconte", "expliquez", "explique", "decrivez", "decris",
    "presentez", "presente", "donnez", "imaginez", "selon vous", "a votre avis", "si vous",
    # anglais
    "what", "why", "how", "when", "where", "which", "who", "can you", "could you", "would you",
    "do you", "did you", "have you", "are you", "is there", "tell me", "walk me", "describe",
    "explain", "give me", "imagine", "suppose",
)

QUESTION_MARKERS = (
    "parlez-moi", "pouvez-vous", "pourriez-vous", "qu'est-ce", "est-ce que", "expliquez", "racontez",
    "comment vous", "comment tu", "quelle est", "quelle serait", "quel est", "la difference entre",
    "tell me", "walk me through", "how would you", "what is", "what's", "difference between",
)


def is_question(text: str) -> bool:
    if "?" in text:
        return True
    t = normalize(text).strip()
    t = re.sub(r"^(alors|donc|et|ok|d'accord|tres bien|bien|bon|so|okay|alright|and)[ ,]+", "", t)
    if t.startswith(QUESTION_STARTERS):
        return True
    return any(m in t for m in QUESTION_MARKERS)


def should_suggest(text: str, auto_all: bool = False) -> bool:
    if len(text.split()) < 3:
        return False
    return auto_all or is_question(text)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def is_echo(me_text: str, me_start: float, me_end: float, them_turns: list, window_s: float = 3.0,
            threshold: float = 0.6) -> bool:
    """Vrai si ce que le micro a capté est en fait la voix de l'interlocuteur sortie des haut-parleurs."""
    for turn in them_turns:
        overlaps = me_start <= turn.end + window_s and me_end >= turn.start - window_s
        if overlaps and (similarity(me_text, turn.text) >= threshold or normalize(me_text) in normalize(turn.text)):
            return True
    return False
