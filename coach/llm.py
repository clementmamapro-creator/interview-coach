"""Moteur Claude : suggestions en direct (streaming), simulation de recruteur, rapport de session."""
from __future__ import annotations

import logging
from typing import AsyncIterator, Protocol

from .config import Settings
from .knowledge import KnowledgeBase

log = logging.getLogger(__name__)

LIVE_INSTRUCTIONS = """Tu es le copilote d'entretien technique de {name}. Tu entends la question posée \
par l'interlocuteur et tu proposes à {name} une réponse qu'il pourra reformuler avec ses mots.

Contraintes de lecture : {name} lit pendant qu'il parle. Tout doit être lisible en 5 secondes.

Format OBLIGATOIRE (markdown, rien d'autre) :
**À dire** : 1 à 2 phrases d'ouverture naturelles, à l'oral.
**Points clés**
- 3 à 5 puces de 12 mots maximum, du plus important au moins important
**Exemple** : une expérience RÉELLE tirée du dossier du candidat (projet, chiffre, outil), en une phrase. \
Si rien de pertinent dans le dossier, écris « — ».

Règles :
- Question technique : réponse exacte et précise (définitions, complexités, commandes, \
compromis), vocabulaire du métier, aucune approximation.
- Question comportementale : structure STAR condensée, appuyée sur le dossier.
- Cas pratique ou question floue : propose d'abord une question de clarification à poser.
- Demande de présentation détaillée d'un projet (« de A à Z », « en détail ») : garde le même format, \
mais avec 5 à 7 puces qui suivent le fil besoin métier → architecture → ce que j'ai fait → difficulté \
résolue → résultat chiffré → suite envisagée.
- N'invente JAMAIS d'expérience, de diplôme ou de chiffre absent du dossier.
- Si le dossier distingue ce qui est « réalisé » de ce qui est une « évolution » ou une « piste », \
présente toujours une piste comme une idée (« la prochaine étape serait… »), jamais comme livrée.
- Utilise les intitulés de poste et les chiffres exactement comme écrits dans le dossier.
- Réponds dans la langue de la question.
{job}"""

SIMULATOR_INSTRUCTIONS = """Tu joues un recruteur technique exigeant qui fait passer un entretien à {name}.
{job}
Pose UNE seule question à la fois, sans préambule ni commentaire, comme à l'oral. Alterne questions \
techniques précises, mises en situation, questions sur les projets du dossier et relances sur la \
réponse précédente si elle était vague. Ne répète pas une question déjà posée. Langue : {lang}."""

REPORT_INSTRUCTIONS = """Tu es un coach d'entretien technique senior. Analyse la transcription d'entretien \
de {name} ci-dessous et rédige un rapport en markdown, en français :

## Synthèse
Note globale /10, 3 forces, 3 axes de progrès prioritaires.
## Analyse question par question
Pour chaque question : la question, ce que {name} a répondu (résumé), note /10, ce qui manquait, \
puis une **meilleure réponse** concise appuyée sur son dossier.
## Notions techniques à réviser
Liste précise.
## Plan d'entraînement
3 exercices concrets pour la prochaine session.

Sois franc et précis : un rapport complaisant ne l'aide pas.
{job}"""


SUMMARY_INSTRUCTIONS = """À partir de la transcription d'un échange entre {name} et son interlocuteur, \
rédige en français, en markdown :

## Compte rendu de l'échange
Date du jour, participants (« {name} » et « l'interlocuteur » si son nom n'est pas cité), objet, \
puis les sujets abordés en 4 à 8 puces factuelles, les points forts mis en avant par {name}, \
les questions ou points ouverts, et les prochaines étapes évoquées.

## Email de suivi
Objet + corps (8 à 12 lignes), ton professionnel et chaleureux, vouvoiement : remerciement, \
2 ou 3 éléments précis de l'échange, rappel de la motivation, prochaine étape. Signé {name}.

Reste strictement fidèle à la transcription : n'ajoute aucun fait, chiffre ou engagement qui n'y figure pas.
{job}"""


def _job_block(job_context: str) -> str:
    return f"\nPoste visé / contexte de l'entretien :\n{job_context.strip()}\n" if job_context.strip() else ""


def format_transcript(turns, last: int | None = None) -> str:
    labels = {"me": "Candidat", "them": "Recruteur"}
    sel = turns[-last:] if last else turns
    return "\n".join(f"{labels.get(t.speaker, t.speaker)} : {t.text}" for t in sel)


class Engine(Protocol):
    def stream_suggestion(self, question: str, turns: list, job_context: str) -> AsyncIterator[str]: ...
    async def simulated_question(self, turns: list, job_context: str) -> str: ...
    async def session_summary(self, turns: list, job_context: str) -> str:
        msg = await self.client.messages.create(
            model=self.s.live_model, max_tokens=2000,
            system=[{"type": "text", "text": SUMMARY_INSTRUCTIONS.format(
                name=self.s.candidate_name, job=_job_block(job_context))}],
            messages=[{"role": "user", "content": f"<transcription>\n{format_transcript(turns)}\n</transcription>"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    async def session_report(self, turns: list, job_context: str) -> str: ...


class ClaudeEngine:
    def __init__(self, settings: Settings, kb: KnowledgeBase) -> None:
        from anthropic import AsyncAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY manquante : renseignez-la dans le fichier .env")
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.s = settings
        self.kb = kb

    # Le dossier candidat est placé dans un bloc système mis en cache (prompt caching) :
    # après la 1re question, il n'est plus retraité -> latence et coût réduits.
    def _use_full_context(self) -> bool:
        return 0 < self.kb.total_chars <= self.s.max_context_chars

    def _system(self, instructions: str) -> list[dict]:
        blocks = [{"type": "text", "text": instructions}]
        if self._use_full_context():
            text = self.kb.full_text()
        elif self.kb.pinned:  # dossier trop gros : on garde en entier les documents prioritaires
            text = self.kb.full_text(only_pinned=True)
        else:
            text = ""
        if text:
            blocks.append({
                "type": "text",
                "text": f"<dossier_candidat>\n{text}\n</dossier_candidat>",
                "cache_control": {"type": "ephemeral"},
            })
        return blocks

    def _rag(self, query: str) -> str:
        if self._use_full_context():
            return ""
        # 1) projet cité nommément -> fiche complète
        named = [n for n in self.kb.match_documents(query) if n not in self.kb.pinned]
        parts = [f"<fiche_complete source=\"{n}\">\n{self.kb.documents[n]}\n</fiche_complete>" for n in named]
        # 2) compléments par recherche BM25
        hits = [c for c in self.kb.search(query, k=self.s.rag_top_k + 6)
                if c.source not in self.kb.pinned and c.source not in named][: self.s.rag_top_k]
        if hits:
            body = "\n\n".join(f"[{c.source}]\n{c.text}" for c in hits)
            parts.append(f"<extraits_dossier>\n{body}\n</extraits_dossier>")
        return ("\n\n".join(parts) + "\n\n") if parts else ""

    async def stream_suggestion(self, question: str, turns: list, job_context: str) -> AsyncIterator[str]:
        system = self._system(LIVE_INSTRUCTIONS.format(name=self.s.candidate_name, job=_job_block(job_context)))
        history = format_transcript(turns, last=10)
        user = (f"{self._rag(question + ' ' + job_context)}"
                f"<conversation_recente>\n{history}\n</conversation_recente>\n\n"
                f"Question à traiter maintenant : « {question} »")
        async with self.client.messages.stream(
            model=self.s.live_model, max_tokens=self.s.max_tokens_live,
            system=system, messages=[{"role": "user", "content": user}],
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def simulated_question(self, turns: list, job_context: str) -> str:
        lang = "anglais" if self.s.language == "en" else "français"
        system = self._system(SIMULATOR_INSTRUCTIONS.format(
            name=self.s.candidate_name, job=_job_block(job_context), lang=lang))
        history = format_transcript(turns, last=16) or "(début de l'entretien)"
        msg = await self.client.messages.create(
            model=self.s.live_model, max_tokens=200, system=system,
            messages=[{"role": "user", "content": f"Transcription jusqu'ici :\n{history}\n\nPose la question suivante."}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    async def session_summary(self, turns: list, job_context: str) -> str:
        msg = await self.client.messages.create(
            model=self.s.live_model, max_tokens=2000,
            system=[{"type": "text", "text": SUMMARY_INSTRUCTIONS.format(
                name=self.s.candidate_name, job=_job_block(job_context))}],
            messages=[{"role": "user", "content": f"<transcription>\n{format_transcript(turns)}\n</transcription>"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    async def session_report(self, turns: list, job_context: str) -> str:
        system = self._system(REPORT_INSTRUCTIONS.format(name=self.s.candidate_name, job=_job_block(job_context)))
        msg = await self.client.messages.create(
            model=self.s.report_model, max_tokens=4000, system=system,
            messages=[{"role": "user", "content": f"<transcription>\n{format_transcript(turns)}\n</transcription>"}],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
