# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""The four seams.

Everything replaceable in this package is replaceable here and nowhere else.
``container.py`` is the only module that names a concrete implementation; every
other module receives what it needs through its constructor.

These are ``typing.Protocol``, not abstract base classes, on purpose. Structural
typing means an implementation never imports anything from this package to
satisfy a seam: a class with the right methods is already a valid ``Embedder``.
That is what keeps ``tests/fakes.py`` to a few dozen lines and what lets a user
drop in their own vector store without inheriting from us.

The two seams that cross the network, :class:`Embedder` and :class:`LLMClient`,
are async all the way up. Upstream builds a synchronous ``openai.OpenAI`` client
at every call site, which inside a LiveKit agent stalls the event loop for the
duration; upstream's own comments measure one conflict-resolution call at 10.2
seconds. Declaring the seam async is what stops ``asyncio.to_thread`` becoming a
permanent fixture: there is nothing synchronous left to wrap.

:class:`GraphStore` is one injectable seam composed of seven narrow protocols.
The container still wires exactly four things, but each vendored module declares
only the slice it actually uses, so a test double for the anchor router is an
:class:`AnchorGraph` and not a seventy-method stand-in for everything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .leftbrain.records import (
        AffectiveEdge,
        Entity,
        EntityEdge,
        SlotProfile,
    )
    from .rightbrain.records import (
        AnchorRole,
        AnchorType,
        MemoryAnchor,
        RightBrainAnchorLink,
        RightBrainMemory,
    )
    from .types import (
        Evidence,
        MemoryHit,
        MemoryRecord,
        Scope,
        StoredTrait,
    )

__all__ = [
    "AnchorGraph",
    "Embedder",
    "EntityGraph",
    "GraphStore",
    "LLMClient",
    "MemoryMeta",
    "SessionState",
    "SlotIndex",
    "SubgraphGraph",
    "TraitGraph",
    "VectorStore",
]


# ---------------------------------------------------------------------------
# Seam 1: embeddings
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors.

    ``embed_query`` and ``embed_documents`` are separate because asymmetric
    models need them to be. E5 requires a ``"query: "`` / ``"passage: "`` prefix
    distinction to reach its real accuracy, and collapsing the two costs
    measurable recall. OpenAI's models treat both identically, so its adapter
    simply routes both to the same call.
    """

    @property
    def model_name(self) -> str:
        """Recorded in ``vm_meta`` so a silent model swap becomes a startup
        error instead of a right brain that returns nothing forever."""
        ...

    @property
    def dimensions(self) -> int:
        """Must match the ``vector(N)`` width the schema was migrated with."""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text that is being stored. Order matches the input."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed text that is being searched with."""
        ...


# ---------------------------------------------------------------------------
# Seam 2: the LLM
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMClient(Protocol):
    """Chat completions, used only on the write path.

    Nothing on the retrieval path calls this. Query classification runs on
    embeddings, which is what keeps recall inside its budget.
    """

    @property
    def model(self) -> str: ...

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        purpose: str = "",
    ) -> dict[str, Any]:
        """Constrained to a JSON object response.

        ``purpose`` is not passed to the model. It labels the call for the
        recorder so the five calls a turn costs can be told apart in the
        latency and cost tables.
        """
        ...

    async def complete_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        purpose: str = "",
    ) -> str: ...


# ---------------------------------------------------------------------------
# Seam 3: the vector store
# ---------------------------------------------------------------------------

TimeExprKind = Literal["date", "duration"]


@runtime_checkable
class VectorStore(Protocol):
    """Facts, their vectors, and ranked retrieval over them.

    One table. Upstream split this across a SQLite metadata row and a Qdrant
    point joined by zipping two id lists together, which is why a write that
    failed its foreign key could be swallowed into a log line.
    """

    async def search(
        self,
        query: str,
        *,
        scope: Scope,
        top_k: int = 10,
        threshold: float | None = None,
        memory_id_filter: Sequence[str] | None = None,
        rescue_k: int = 0,
        include_assistant: bool = False,
        query_vector: Sequence[float] | None = None,
    ) -> list[MemoryHit]:
        """Ranked retrieval. The scoring contract is fixed and must be preserved:

        1. Rows authored by the assistant are excluded unless
           ``include_assistant``. Without this the agent retrieves and quotes
           its own past replies, and answers drift further every turn.
        2. ``base_score`` is the raw cosine. ``score`` adds the lexical and
           date-overlap bonuses.
        3. Results are ordered by ``base_score``, not ``score``, then cut to
           ``top_k``. Bonuses must not displace what was semantically relevant.
        4. Up to ``rescue_k`` further hits with ``time_boost`` set are appended
           after that cut, ordered by ``score``. This is how a time-relevant
           memory that ranked poorly on cosine is recovered.
        5. ``memory_id_filter`` restricts candidates and must be applied in SQL.
           Upstream fetched 10,000 rows and filtered in Python.

        ``query_vector`` lets a caller that has already embedded this text avoid
        a second round trip. Recall passes one, because slot classification,
        ranking and trait search all need the same vector and embedding it three
        times measured at roughly 200ms each.
        """
        ...

    async def add_records(self, scope: Scope, records: Sequence[MemoryRecord]) -> list[str]:
        """Store facts, returning their ids in input order."""
        ...

    async def add_text(
        self,
        scope: Scope,
        text: str,
        *,
        attributed_to: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store one row verbatim, bypassing extraction.

        Used for the agent's own replies, which are stored with an assistant
        role so ``search`` can exclude them by default.
        """
        ...

    async def update_memory(
        self,
        scope: Scope,
        memory_id: str,
        new_text: str,
        *,
        session_id: str | None = None,
        observed_at: str | None = None,
    ) -> bool: ...

    async def delete_memory(self, scope: Scope, memory_id: str) -> bool: ...

    async def archive_memory(self, scope: Scope, memory_id: str) -> bool:
        """Hide a cold memory from retrieval without deleting it."""
        ...

    async def unarchive_memory(self, scope: Scope, memory_id: str) -> bool: ...

    async def list_ids(self, scope: Scope) -> list[str]: ...

    async def list_entries(self, scope: Scope) -> list[MemoryRecord]:
        """Every memory, for batch work such as refreshing slot summaries."""
        ...

    async def existing_for_extractor(self, scope: Scope, *, limit: int = 50) -> list[MemoryRecord]:
        """Candidates shown to the conflict resolver so it can decide
        ADD, UPDATE, DELETE or NONE against what is already known."""
        ...

    async def memory_ids_with_time_expr(
        self, scope: Scope, *, kind: TimeExprKind
    ) -> set[str]:
        """Ids whose text contains a date or a duration expression.

        Widens the candidate pool for "when" and "how long" questions, whose
        answers frequently rank below the cosine threshold.
        """
        ...


# ---------------------------------------------------------------------------
# Seam 4: the graph, as seven narrow protocols
# ---------------------------------------------------------------------------


class EntityGraph(Protocol):
    """People, places and things, and how they relate."""

    async def upsert_entity(
        self,
        scope: Scope,
        *,
        name: str,
        entity_type: str,
        slot: str,
        description: str = "",
        embedding: Sequence[float] | None = None,
    ) -> str:
        """Create or merge an entity, returning its id.

        Deduplication is semantic, not exact. Upstream loaded every entity of
        the type into Python and compared cosines one at a time on each mention;
        this must be a single nearest-neighbour query.
        """
        ...

    async def get_entity(self, scope: Scope, entity_id: str) -> Entity | None: ...

    async def find_entities(
        self, scope: Scope, *, names: Sequence[str] | None = None, slot: str | None = None
    ) -> list[Entity]: ...

    async def find_entities_by_name_fuzzy(
        self, scope: Scope, name: str, *, limit: int = 5
    ) -> list[Entity]: ...

    async def upsert_edge(
        self,
        scope: Scope,
        *,
        from_entity_id: str,
        to_entity_id: str,
        relation_type: str,
        evidence_memory_id: str | None = None,
    ) -> str:
        """Relate two entities. Repeated co-occurrence accumulates weight, which
        decays with time on read."""
        ...

    async def edges_for_entity(self, scope: Scope, entity_id: str) -> list[EntityEdge]: ...

    async def neighbor_entity_ids(
        self, scope: Scope, entity_ids: Sequence[str]
    ) -> list[str]:
        """One hop out. This is what turns a question about a person into the
        projects and places attached to them."""
        ...

    async def entity_context(
        self, scope: Scope, entity_id: str, *, depth: int = 1
    ) -> dict[str, Any]:
        """An entity with its edges, linked memory ids and affective edges."""
        ...

    async def upsert_affective_edge(
        self, scope: Scope, *, entity_id: str, valence: float, arousal: float, label: str = ""
    ) -> str:
        """How the user feels about an entity, as opposed to what is true of it."""
        ...

    async def affective_edges_for_entity(
        self, scope: Scope, entity_id: str
    ) -> list[AffectiveEdge]: ...


class MemoryMeta(Protocol):
    """The graph's side of a memory: which entities and slots it belongs to."""

    async def link_memory(
        self,
        scope: Scope,
        *,
        memory_id: str,
        entity_id: str,
        role: str = "context",
        relation_hint: str | None = None,
    ) -> None: ...

    async def memory_ids_for_entities(
        self, scope: Scope, entity_ids: Sequence[str]
    ) -> list[str]: ...

    async def memory_ids_for_slots(self, scope: Scope, slots: Sequence[str]) -> list[str]: ...

    async def entity_ids_for_memory(self, scope: Scope, memory_id: str) -> list[str]: ...

    async def all_linked_memory_ids(self, scope: Scope) -> list[str]: ...

    async def record_memory_hits(self, scope: Scope, memory_ids: Sequence[str]) -> None:
        """Mark memories as retrieved. Feeds the heat score that decides what
        eventually gets archived."""
        ...

    async def get_memory_heat(self, scope: Scope, memory_id: str) -> float | None: ...

    async def list_archivable_memories(
        self, scope: Scope, *, older_than_days: int, max_heat: float
    ) -> list[str]: ...

    async def ingest_annotated_fact(
        self,
        scope: Scope,
        *,
        memory_id: str,
        slot: str,
        entities: Sequence[str],
        relations: Sequence[tuple[str, str, str]],
    ) -> None:
        """Write one annotated fact into the graph: upsert its entities, link
        them to the memory, and record the relations between them."""
        ...


class SlotIndex(Protocol):
    """Life-domain routing. Which of the seven slots a memory belongs to, and
    the summaries that describe each slot's contents."""

    async def upsert_memory_tags(
        self, scope: Scope, memory_id: str, tags: Sequence[tuple[str, float]]
    ) -> None: ...

    async def memory_ids_for_slots_v2(
        self, scope: Scope, slots: Sequence[str], *, min_confidence: float = 0.0
    ) -> list[str]: ...

    async def get_tags_for_memory(
        self, scope: Scope, memory_id: str
    ) -> list[tuple[str, float]]: ...

    async def memory_tag_counts(self, scope: Scope) -> dict[str, int]: ...

    async def count_memories_in_slot(self, scope: Scope, slot: str) -> int: ...

    async def upsert_slot_summary(self, scope: Scope, slot: str, summary: str) -> None: ...

    async def get_slot_summaries(
        self, scope: Scope, slots: Sequence[str]
    ) -> dict[str, str]: ...

    async def record_slot_cooccurrence(self, scope: Scope, slots: Sequence[str]) -> None:
        """Note that these slots were activated together, which is how the
        related-slot edges emerge from use rather than being declared."""
        ...

    async def get_macro_related_slots(self, scope: Scope, slot: str, *, limit: int = 3) -> list[str]: ...

    async def refresh_slot_profile(self, scope: Scope, slot: str) -> None: ...

    async def slot_profiles(self, scope: Scope) -> list[SlotProfile]: ...


class SubgraphGraph(Protocol):
    """Emergent sub-slots.

    When one slot accumulates enough densely connected activity, it splits into
    a named child. Off by default: the check costs LLM calls at session
    boundaries.
    """

    async def record_query_activation(
        self, scope: Scope, *, entity_ids: Sequence[str], session_id: str
    ) -> None: ...

    async def compute_rho(self, scope: Scope, entity_ids: Sequence[str]) -> float:
        """Co-activation density for a candidate cluster."""
        ...

    async def get_or_create_entity_semantic(
        self,
        scope: Scope,
        *,
        name: str,
        slot_ref: str,
        embedding: Sequence[float],
        threshold: float = 0.86,
    ) -> str: ...

    async def get_entities_for_slot(self, scope: Scope, slot_ref: str) -> list[Any]: ...

    async def get_entities_for_memory(self, scope: Scope, memory_id: str) -> list[Any]: ...

    async def update_slot_ref(self, scope: Scope, entity_id: str, new_slot_ref: str) -> None: ...

    async def set_description(self, scope: Scope, entity_id: str, description: str) -> None: ...

    async def mark_cannot_split(self, scope: Scope, entity_id: str) -> None: ...

    async def link_graph_memory(self, scope: Scope, entity_id: str, memory_id: str) -> None: ...

    async def get_dynamic_slots(self, scope: Scope) -> list[Any]: ...

    async def create_dynamic_slot(
        self,
        scope: Scope,
        *,
        name: str,
        description: str,
        embedding: Sequence[float] | None,
        parent_slots: Sequence[str],
    ) -> None: ...

    async def get_children(self, scope: Scope, parent_name: str) -> list[Any]: ...

    async def dynamic_slot_exists(self, scope: Scope, name: str) -> bool: ...


class AnchorGraph(Protocol):
    """Right-brain memories and the anchors they hang from.

    An anchor is what makes a persona note retrievable: an emotion, an entity,
    or a global marker. Upstream measured that essentially every note anchors on
    emotion, which is why the emotion label matters so much on the write path.
    """

    async def upsert_memory(
        self,
        scope: Scope,
        *,
        memory_class: str,
        content: str,
        priority: float,
        confidence: float,
        ttl: str,
        condition: str | None = None,
        metadata: dict[str, Any] | None = None,
        evidence_memory_ids: Sequence[str] = (),
    ) -> str: ...

    async def link_anchor(
        self,
        scope: Scope,
        *,
        right_memory_id: str,
        anchor_type: AnchorType,
        anchor_id: str | None,
        role: AnchorRole,
        weight: float = 1.0,
        confidence: float = 1.0,
    ) -> None: ...

    async def search_by_anchors(
        self, scope: Scope, anchors: Sequence[MemoryAnchor], *, top_k: int = 10
    ) -> list[RightBrainMemory]:
        """Retrieval here is 0 LLM calls and runs concurrently with the left
        brain's ranking, so it adds no wall-clock to a turn."""
        ...

    async def search_global(self, scope: Scope, *, top_k: int = 5) -> list[RightBrainMemory]:
        """Notes with no specific anchor: the standing profile."""
        ...

    async def get_memory(self, scope: Scope, memory_id: str) -> RightBrainMemory | None: ...

    async def update_content(self, scope: Scope, memory_id: str, content: str) -> None: ...

    async def merge_metadata(
        self, scope: Scope, memory_id: str, updates: dict[str, Any]
    ) -> None:
        """Must be a single statement. Upstream read, merged in Python and wrote
        back, which loses concurrent updates."""
        ...

    async def get_anchors_for_memory(
        self, scope: Scope, right_memory_id: str
    ) -> list[RightBrainAnchorLink]: ...


class TraitGraph(Protocol):
    """Judgements about what the user is like, with the evidence behind them."""

    async def add_trait(
        self, scope: Scope, *, slot: str, claim: str, evidence: Evidence,
        embedding: Sequence[float] | None = None,
    ) -> str: ...

    async def all_traits(self, scope: Scope, *, per_slot: int = 8) -> list[StoredTrait]: ...

    async def search_traits(
        self, scope: Scope, query: str, *, embedding: Sequence[float], top_k: int = 5
    ) -> list[StoredTrait]:
        """Retrieved against the query rather than returned wholesale.

        Upstream previously returned the same static profile lines on every
        turn regardless of what was asked, which reads as a dossier rather than
        as memory.
        """
        ...

    async def trait_counts(self, scope: Scope) -> tuple[int, int]: ...


class SessionState(Protocol):
    """Conversation boundaries and the deferred work they trigger."""

    async def record_turn(self, scope: Scope, session_id: str) -> dict[str, Any]:
        """Register a turn, returning whether this opened a new session."""
        ...

    async def get_current_session(self, scope: Scope) -> str | None: ...

    async def touch(self, scope: Scope, namespace: str, ref: str) -> None:
        """Mark something as changed this session, for batch work at the end."""
        ...

    async def count_touched(self, scope: Scope, namespace: str) -> int: ...

    async def pop_touched(self, scope: Scope, namespace: str) -> list[str]:
        """Take and clear. Must be a single ``DELETE ... RETURNING``; upstream's
        select-then-delete double-processes under concurrent ingests."""
        ...


class GraphStore(
    EntityGraph,
    MemoryMeta,
    SlotIndex,
    SubgraphGraph,
    AnchorGraph,
    TraitGraph,
    SessionState,
    Protocol,
):
    """Everything relational, as one injectable seam.

    The container wires this once. Individual modules should depend on the
    narrow protocol they actually use, not on this composition.
    """

    async def delete_user(self, scope: Scope) -> None:
        """Erase everything for one user. This is the deletion request path, so
        it must cover all tables, not only the ones a caller remembers."""
        ...
