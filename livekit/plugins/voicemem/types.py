# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Portions derived from VoiceMem (https://github.com/xzf-thu/VoiceMem), Apache-2.0.
#   MemoryHit          <- voicemem/leftbrain/local_memory_store.py MemorySearchHit
#   QueryClassification<- voicemem/leftbrain/cognitive_graph/query_slot_classifier.py
#   RightBrainHit      <- voicemem/rightbrain/brain.py RightBrainHit
# See CHANGES-FROM-UPSTREAM.md.
"""Value objects passed between the brains, the stores and the hook layer.

Everything here is frozen. The one piece of upstream state that was not a value
object, the ``threading.local()`` scratchpad in ``merged_extraction.py`` used to
smuggle emotion and traits from the extractor to the right brain, is replaced by
:class:`Extraction`: one object, returned by one call, handed explicitly to both
brains. Under asyncio a single thread serves every session, so the upstream
design leaked one user's emotional state into another user's write.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "Annotation",
    "Evidence",
    "Extraction",
    "Fact",
    "MemoryHit",
    "MemoryRecord",
    "QueryClassification",
    "RecallResult",
    "Resolution",
    "ResolutionAction",
    "RightBrainHit",
    "Scope",
    "StoredTrait",
    "Trait",
    "TurnRecord",
]

Role = Literal["user", "assistant"]


@dataclass(frozen=True, slots=True)
class Scope:
    """Who a storage operation is for.

    Upstream passes ``user_id`` alone, because its tenancy boundary was the
    SQLite file: one file per memory space meant no table ever needed a tenant
    column. In one shared Postgres that boundary does not exist, so every query
    carries both halves. Bundling them means the pair cannot be split up by
    accident on the way through seventy store methods, and a missing tenant is a
    type error rather than a cross-tenant read.
    """

    tenant_id: str
    user_id: str

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id must not be empty")
        if not self.user_id:
            raise ValueError("user_id must not be empty")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MemoryHit:
    """One retrieved fact.

    ``score`` is the ranking score: cosine plus the lexical and date bonuses.
    ``base_score`` is the raw cosine before any bonus. Upstream keeps both so
    that bonuses can rescue a buried memory without displacing the ones that
    were semantically relevant to begin with, and the final ordering is by
    ``base_score``. Preserving that distinction is what keeps our ranking
    diffable against upstream.
    """

    memory_id: str
    text: str
    score: float
    attributed_to: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)
    base_score: float = 0.0
    #: This hit matched the *kind* of time the question asked about: a duration
    #: expression for a "how long" question, or a date for a "when" question.
    time_boost: bool = False
    #: The day the remembered event happened (YYYY-MM-DD), from the turn's
    #: observed_at. Fact text usually says "last week" rather than a date, so
    #: without this, ordering two memories in time has nothing to work from.
    observed_at: str = ""


@dataclass(frozen=True, slots=True)
class RightBrainHit:
    """One piece of persona or emotional context.

    This is an internal note for the model, never something the agent may say
    back to the user. The renderer labels it accordingly.
    """

    content: str
    source: str
    priority: float = 0.0
    observed_at: str = ""


@dataclass(frozen=True, slots=True)
class QueryClassification:
    """What a query is about: which slots, and which named entities."""

    slots: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    #: Set when the classifier drilled into a dynamic child slot.
    child_slots: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecallResult:
    """Everything one recall produced, including what it cost.

    ``block`` is the rendered prompt text and is empty when there is nothing
    worth injecting, which is the common case early in a conversation.
    """

    block: str
    hits: tuple[MemoryHit, ...] = ()
    rb_hits: tuple[RightBrainHit, ...] = ()
    #: Per-stage milliseconds. Keys mirror upstream's timing dict so the numbers
    #: stay comparable: slot_filter, entity_narrow, rank, rb, total.
    timing: dict[str, float] = field(default_factory=dict)
    #: True when this result came from a prefetch started on an interim
    #: transcript rather than a live lookup at end of turn.
    prefetched: bool = False


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Annotation:
    """Slot, entity and relation labels for one extracted fact."""

    slot: str = ""
    entities: tuple[str, ...] = ()
    relations: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Fact:
    """One atomic statement extracted from a turn."""

    text: str
    annotation: Annotation = field(default_factory=Annotation)
    #: Whoever the fact is about, as the extractor labelled it.
    attributed_to: str = "user"


@dataclass(frozen=True, slots=True)
class Trait:
    """One observation about what the user is like, as opposed to what is true.

    Upstream calls these right-brain traits: preferences, habits and ways of
    thinking that no fact captures. "I hate loud chewing" is a trait, not a fact.
    """

    slot: str
    label: str


@dataclass(frozen=True, slots=True)
class Extraction:
    """The whole result of the one extraction call, for both brains.

    Replaces ``voicemem/leftbrain/merged_extraction.py`` in full. Upstream
    returned facts from the call and stashed ``emotion`` and ``traits`` in
    module-level dicts and a ``threading.local()`` for other components to pick
    up later. That is a cross-user data leak under asyncio and it also broke on
    a key mismatch, because the extractor's copy of the utterance carried a
    "Speaker 0: " prefix the right brain's did not.
    """

    facts: tuple[Fact, ...] = ()
    #: Emotion judged from the text of the turn. Upstream prefers this over its
    #: own acoustic detector, whose keyword table scored "I'm so angry, my boss
    #: keeps pressuring me" as anxiety because it matched on "pressure".
    emotion: str = ""
    traits: tuple[Trait, ...] = ()
    #: True when the single merged call supplied annotations, so the separate
    #: annotator call can be skipped. False means the merge did not parse and
    #: the caller must fall back.
    merged: bool = False


ResolutionAction = Literal["ADD", "UPDATE", "DELETE", "NONE"]


@dataclass(frozen=True, slots=True)
class Resolution:
    """What the conflict resolver decided about one candidate fact."""

    action: ResolutionAction
    text: str
    memory_id: str | None = None


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One conversational turn, queued for background ingestion.

    Carries the agent's reply because both brains need it: the left brain uses
    it to disambiguate what the user meant, and the right brain attributes the
    user's reaction to it. The same "fine" means different things after an
    apology and after a refusal.
    """

    tenant_id: str
    user_id: str
    user_text: str
    agent_reply: str = ""
    #: Reply of the turn *before* this one, which is what the user is reacting
    #: to. Distinct from agent_reply, which answers this turn.
    prior_agent_reply: str = ""
    #: When the remembered event happened, ISO date. Backfilling history must
    #: set this explicitly or every memory looks like it happened today.
    observed_at: str = ""
    session_id: str = ""
    #: True when the agent was cut off mid-reply, so agent_reply holds only
    #: what the user actually heard.
    interrupted: bool = False


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evidence:
    """Why the system believes a trait. Derived from VoiceMem's traits_store."""

    quote: str
    emotion: str = ""
    #: The left-brain fact behind the claim, rendered as the "why".
    cause: str = ""
    cause_id: str = ""
    at: str = ""


@dataclass(frozen=True, slots=True)
class StoredTrait:
    """A persisted judgement about the user, as the trait store returns it.

    Distinct from :class:`Trait`, which is what one extraction call produced and
    has not been reconciled with anything yet.
    """

    trait_id: str
    slot: str
    claim: str
    confidence: float = 0.9
    evidence: tuple[Evidence, ...] = ()
    updated_at: str = ""
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """A row of the merged ``memories`` table, as the store reads and writes it.

    Upstream split this across two systems, a SQLite metadata row and a Qdrant
    point, joined by zipping two id lists together. One table, one id.
    """

    memory_id: str
    text: str
    role: Role = "user"
    attributed_to: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_at: str = ""
    slot: str = ""
