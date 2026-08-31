# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Derived from VoiceMem (https://github.com/xzf-thu/VoiceMem), Apache-2.0,
# commit d587d424a02727d9ff3ef6f8e672a39a19ce64a7, files:
#   voicemem/leftbrain/local_memory_store.py  (the bonus functions and regexes)
#   voicemem/leftbrain/mem0_backend_store.py:366-440  (the ordering contract)
# Changes: comments translated to English; the two halves gathered into one pure
#   module with no I/O so the scoring can be tested on its own and shared by the
#   real store and the in-memory fake. ENGLISH DATE MATCHING ADDED: upstream's
#   query_dates only recognised Chinese date literals, so the English date
#   expansion added in timeexpand.py would have produced dates this bonus could
#   never match. Weights and ordering are unchanged.
# See CHANGES-FROM-UPSTREAM.md.
"""Scoring, ordering and the rescue pass. No I/O, no state, no async.

The contract this module implements is the reason retrieval quality is
comparable to upstream's. Two parts matter and are easy to get wrong:

* ``score`` and ``base_score`` are different numbers. ``base_score`` is the raw
  cosine; ``score`` adds lexical overlap and date bonuses. The final ordering
  uses ``base_score``, so a bonus can never displace something that was
  semantically relevant to begin with.
* The bonuses exist to *rescue*, not to rank. After the top-k cut, a small
  number of extra hits that matched the question's time type are appended. That
  is how "what do I have next week" finds a diary entry whose cosine was poor.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from ..types import MemoryHit

__all__ = [
    "content_words",
    "date_overlap_bonus",
    "lexical_time_bonus",
    "query_dates",
    "rank_hits",
    "time_question_kind",
]

# ---------------------------------------------------------------------------
# Recognisers
# ---------------------------------------------------------------------------

#: A date appearing in stored memory text. The extractor normalises relative
#: expressions to absolute dates, in "August 26, 2026" for English and
#: "2026年8月26日" for Chinese.
_DATE_RE = re.compile(
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(?:19|20)\d{2}\b"
    r"|\b(?:19|20)\d{2}-\d{2}-\d{2}\b"
    r"|\b(?:last|next)\s+(?:year|month|week)\b"
    r"|(?:(?:19|20)\d{2}年)?\d{1,2}月\d{1,2}[日号]",
    re.I,
)

_DURATION_RE = re.compile(
    r"\b\d+\s*(?:year|month|week|day|hour|minute)s?\b"
    r"|\d+\s*(?:年|个月|周|天|小时|分钟)",
    re.I,
)

_DURATION_Q_RE = re.compile(
    r"\bhow\s+long\b|\bhow\s+many\s+(?:year|month|week|day|hour)s?\b|多久|多长时间", re.I
)

#: Whether the question is asking *when*. The Chinese relative-time words are
#: here because a question like "我下周有什么安排" is a time question without
#: containing the word "when"; the English equivalents are here for the same
#: reason.
_DATE_Q_RE = re.compile(
    r"\bwhen\b|\bwhat\s+(?:date|day|time)\b"
    r"|\b(?:yesterday|today|tomorrow)\b"
    r"|\b(?:last|next|this|coming)\s+(?:week|month|year)\b"
    r"|\bschedule\b|\bplans?\b|\bagenda\b|\bcalendar\b"
    r"|哪天|什么时候"
    r"|今天|明天|后天|昨天|前天|这周|本周|下周|下星期|下个星期|上周|上个星期"
    r"|接下来|这几天|安排|日程|行程",
    re.I,
)

#: Date literals as they appear in a query, in both the formats
#: ``timeexpand.expand_relative_dates`` emits. Most queries contain none of
#: their own; these are almost always the ones expansion appended.
_DATE_LITERAL_RE = re.compile(
    r"(?:(?:19|20)\d{2}年)?\d{1,2}月\d{1,2}日"
    r"|(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s*(?:19|20)\d{2}",
)

#: High-frequency words with no discriminating power. Including them in the
#: overlap ratio only adds noise.
_STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at for from by with about as into over
    is are was were be been being do does did have has had can could will would
    shall should may might must what when where who whom which why how many much
    long time you your their his her its it they them he she we us our i me my
    that this these those there here not no yes so than then get got make made
    """.split()
)

#: Cosine here runs roughly 0.2 to 0.6, which is the scale these are tuned to.
_LEX_WEIGHT = 0.15
_TIME_WEIGHT = 0.10
_DATE_MATCH_WEIGHT = 0.12


def content_words(text: str) -> set[str]:
    """Lowercase content words of three characters or more."""
    return {
        w for w in re.findall(r"[a-z0-9']+", text.lower())
        if len(w) >= 3 and w not in _STOPWORDS
    }


def time_question_kind(query: str) -> str | None:
    """``"duration"``, ``"date"``, or ``None`` if this is not a time question."""
    if _DURATION_Q_RE.search(query):
        return "duration"
    if _DATE_Q_RE.search(query):
        return "date"
    return None


def query_dates(query: str) -> frozenset[str]:
    """Date literals present in the query, in either supported format."""
    return frozenset(_DATE_LITERAL_RE.findall(query or ""))


def date_overlap_bonus(q_dates: frozenset[str], mem_text: str) -> float:
    """Whether the memory's own date falls inside the days the query covers.

    Cosine distinguishes this badly. Upstream measured "what do I have next
    week" surfacing only two of three diary entries, because a semantically
    closer entry from the wrong week outranked one from the right week. This is
    an exact string comparison, so what it adds is the hard fact of being in
    range rather than more similarity.
    """
    if not q_dates:
        return 0.0
    return _DATE_MATCH_WEIGHT if any(d in mem_text for d in q_dates) else 0.0


def lexical_time_bonus(
    q_words: set[str], *, want_duration: bool, want_date: bool, mem_text: str
) -> tuple[float, bool]:
    """``(bonus, matched_the_questions_time_type)`` for one memory."""
    if not q_words:
        return 0.0, False
    overlap = len(q_words & content_words(mem_text)) / len(q_words)
    bonus = _LEX_WEIGHT * overlap
    # The time bonus is gated on some lexical overlap. Without that gate, every
    # memory in the store that happens to contain a date gets lifted.
    time_hit = bool(
        overlap > 0
        and (
            (want_duration and _DURATION_RE.search(mem_text))
            or (want_date and _DATE_RE.search(mem_text))
        )
    )
    if time_hit:
        bonus += _TIME_WEIGHT
    return bonus, time_hit


def rank_hits(
    query: str,
    candidates: Iterable[tuple[str, str, float, str, dict, str]],
    *,
    top_k: int,
    rescue_k: int = 0,
    threshold: float | None = None,
) -> list[MemoryHit]:
    """Apply the full scoring contract to raw candidates.

    ``candidates`` are ``(memory_id, text, cosine, attributed_to, metadata,
    observed_at)`` tuples, which is what both the Postgres store and the
    in-memory fake produce. Sharing this function is what stops their scoring
    from drifting apart.
    """
    kind = time_question_kind(query)
    want_duration = kind == "duration"
    want_date = kind == "date"
    q_words = content_words(query)
    q_dates = query_dates(query)

    hits: list[MemoryHit] = []
    for memory_id, text, cosine, attributed_to, metadata, observed_at in candidates:
        if threshold is not None and cosine < threshold:
            continue
        bonus, time_hit = lexical_time_bonus(
            q_words, want_duration=want_duration, want_date=want_date, mem_text=text
        )
        bonus += date_overlap_bonus(q_dates, text)
        hits.append(
            MemoryHit(
                memory_id=memory_id,
                text=text,
                score=cosine + bonus,
                attributed_to=attributed_to or "user",
                metadata=dict(metadata or {}),
                base_score=cosine,
                time_boost=time_hit,
                observed_at=observed_at or "",
            )
        )

    # Ordered by raw cosine, deliberately. Bonuses rescue; they do not rank.
    hits.sort(key=lambda h: h.base_score, reverse=True)
    base = hits[:top_k]
    if rescue_k <= 0:
        return base

    seen = {h.memory_id for h in base}
    boosted = sorted(
        (h for h in hits if h.time_boost and h.memory_id not in seen),
        key=lambda h: h.score,
        reverse=True,
    )
    return base + boosted[:rescue_k]


def dedupe_near(hits: Sequence[MemoryHit], *, threshold: float = 0.82) -> list[MemoryHit]:
    """Drop near-duplicate texts, keeping the higher-scoring one.

    Trigram Jaccard rather than embeddings: this runs inside the turn budget and
    a second round of vector work would not pay for itself.
    """

    def trigrams(text: str) -> set[str]:
        t = re.sub(r"\s+", "", text.lower())
        return {t[i : i + 3] for i in range(max(len(t) - 2, 0))} or {t}

    out: list[MemoryHit] = []
    kept: list[set[str]] = []
    for hit in hits:
        g = trigrams(hit.text)
        if any(len(g & k) / max(len(g | k), 1) >= threshold for k in kept):
            continue
        out.append(hit)
        kept.append(g)
    return out
