# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Derived from VoiceMem (https://github.com/xzf-thu/VoiceMem), Apache-2.0,
# commit d587d424a02727d9ff3ef6f8e672a39a19ce64a7, file:
#   voicemem/leftbrain/extract_facts_openai.py
# Changes: async; the module-level _MERGED_UTTERANCE dict and the threading.local
#   scratchpad are gone, replaced by returning one Extraction value object;
#   comments translated. Junk filtering, the request-clause stripper and the
#   index-based conflict mapping are ported behaviour-for-behaviour.
# See CHANGES-FROM-UPSTREAM.md.
"""Turning an utterance into facts, traits and an emotion, in one call."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..log import logger
from ..types import Annotation, Extraction, Fact, Resolution, Trait
from .prompts import (
    TRAIT_SLOTS,
    build_conflict_user_message,
    build_extraction_user_message,
    extraction_system_prompt,
)

if TYPE_CHECKING:
    from ..protocols import LLMClient

__all__ = ["ConflictResolver", "Extractor", "is_junk", "strip_request_clause"]

#: Text that carries no memory even when the model returns it. Upstream keeps an
#: equivalent list; these are the failure cases worth guarding, not style rules.
_JUNK_RE = re.compile(
    r"^\s*(?:"
    r"the user (?:said|asked|wants to know|is asking|greeted|responded)"
    r"|user (?:said|asked|greeted)"
    r"|(?:hi|hello|hey|thanks|thank you|ok|okay|sure|yes|no)\b\W*$"
    r"|no (?:new )?(?:memory|information|facts?)\b"
    r")",
    re.I,
)

#: A trailing ask, which is not a memory. "I'm taking the GRE next week, any book
#: recommendations?" should store the exam and drop the request.
_REQUEST_CLAUSE_RE = re.compile(
    r"\s*(?:,|and|;)?\s*(?:and )?(?:the user )?(?:is )?"
    r"(?:asking|asks|requests?|wants|would like|looking)\s+(?:for|to|if|whether)\b.*$",
    re.I,
)


def is_junk(text: str) -> bool:
    """Whether an extracted line is worth storing at all."""
    t = (text or "").strip()
    return not t or len(t) < 8 or bool(_JUNK_RE.match(t))


def strip_request_clause(text: str) -> str:
    """Drop a trailing request while keeping the lasting half of the sentence."""
    stripped = _REQUEST_CLAUSE_RE.sub("", text or "").strip().rstrip(",;")
    # Never let the stripper eat the whole sentence.
    return stripped if len(stripped) >= 8 else (text or "").strip()


def _parse_annotation(raw: dict[str, Any]) -> Annotation:
    entities = tuple(
        str(e.get("name", "")).strip()
        for e in (raw.get("entities") or [])
        if isinstance(e, dict) and str(e.get("name", "")).strip()
    )
    relations = tuple(
        (str(r.get("from", "")), str(r.get("relation_type", "")), str(r.get("to", "")))
        for r in (raw.get("relations") or [])
        if isinstance(r, dict) and r.get("from") and r.get("to")
    )
    return Annotation(slot=str(raw.get("slot", "") or ""), entities=entities, relations=relations)


class Extractor:
    """One LLM call producing facts, their annotations, an emotion and traits.

    Upstream splits this across three calls and smuggles two of the results
    between components through module state. Here the caller receives one frozen
    :class:`~livekit.plugins.voicemem.types.Extraction` and passes it explicitly
    to whichever brain needs it.
    """

    __slots__ = ("_llm", "_merged")

    def __init__(self, llm: LLMClient, *, merged: bool = True) -> None:
        self._llm = llm
        self._merged = merged

    async def extract(
        self,
        *,
        user_text: str,
        agent_reply: str = "",
        existing: list[dict[str, str]] | None = None,
        observation_date: str | None = None,
    ) -> Extraction:
        text = (user_text or "").strip()
        if not text:
            return Extraction()

        raw = await self._llm.complete_json(
            system=extraction_system_prompt(merged=self._merged),
            user=build_extraction_user_message(
                user_text=text,
                agent_reply=agent_reply,
                existing=existing,
                observation_date=observation_date,
            ),
            temperature=0.0,
            purpose="extract",
        )

        items = raw.get("memory")
        if not isinstance(items, list):
            logger.warning("voicemem: extraction returned no 'memory' array")
            return Extraction()

        facts: list[Fact] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            body = strip_request_clause(str(item.get("text", "")))
            if is_junk(body):
                continue
            facts.append(
                Fact(
                    text=body,
                    annotation=_parse_annotation(item),
                    attributed_to="user",
                )
            )

        emotion = str(raw.get("emotion", "") or "").strip().lower()
        traits: list[Trait] = []
        for t in raw.get("traits") or []:
            if not isinstance(t, dict):
                continue
            slot = str(t.get("slot", "")).strip()
            label = str(t.get("label", "")).strip()
            # A slot outside the vocabulary means the model improvised; storing
            # it would fragment the trait space silently.
            if label and slot in TRAIT_SLOTS:
                traits.append(Trait(slot=slot, label=label))

        # merged is True only when the response actually carried the extra keys,
        # so a model that ignored the addendum makes the caller fall back rather
        # than silently losing the right brain.
        merged_ok = self._merged and ("traits" in raw or "emotion" in raw)
        return Extraction(
            facts=tuple(facts), emotion=emotion, traits=tuple(traits), merged=merged_ok
        )


class ConflictResolver:
    """Decides ADD, UPDATE, DELETE or NONE for each candidate fact.

    Kept rather than delegated. mem0's own ``infer=True`` path does exact
    hash-based deduplication only, so anything not byte-identical comes back as
    ADD; this is what stops a store filling with near-duplicate restatements of
    the same fact.
    """

    __slots__ = ("_llm",)

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def resolve(
        self, *, existing: list[dict[str, Any]], new_facts: list[str]
    ) -> list[Resolution]:
        if not new_facts:
            return []
        if not existing:
            # Nothing to conflict with. Upstream skips the call here too, and it
            # is the single most expensive call on the write path.
            return [Resolution(action="ADD", text=t) for t in new_facts]

        # The whole prompt goes in the user message; see build_conflict_user_message
        # for why splitting it across roles produces a 400 from json_object mode.
        raw = await self._llm.complete_json(
            system="You update a memory store. Return only a JSON object.",
            user=build_conflict_user_message(existing=existing, new_facts=new_facts),
            temperature=0.0,
            purpose="conflict",
        )

        items = raw.get("memory")
        if not isinstance(items, list):
            logger.warning("voicemem: conflict resolution unparseable, defaulting to ADD")
            return [Resolution(action="ADD", text=t) for t in new_facts]

        out: list[Resolution] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            event = str(item.get("event", "")).upper()
            if event not in ("ADD", "UPDATE", "DELETE", "NONE"):
                continue
            body = str(item.get("text", "")).strip()

            memory_id: str | None = None
            if event in ("UPDATE", "DELETE"):
                # The model was shown integer indices, not UUIDs, so an invented
                # id is out of range rather than plausible.
                try:
                    idx = int(str(item.get("id", "")))
                except (TypeError, ValueError):
                    continue
                if not 0 <= idx < len(existing):
                    logger.warning(
                        "voicemem: conflict resolver referenced index %s of %d; dropping",
                        idx, len(existing),
                    )
                    continue
                memory_id = str(existing[idx].get("id", "")) or None
                if memory_id is None:
                    continue

            if event in ("ADD", "UPDATE") and is_junk(body):
                continue
            out.append(Resolution(action=event, text=body, memory_id=memory_id))
        return out
