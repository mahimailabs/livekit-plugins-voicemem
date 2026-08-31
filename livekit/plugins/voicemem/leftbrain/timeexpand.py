# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Derived from VoiceMem (https://github.com/xzf-thu/VoiceMem), Apache-2.0,
# commit d587d424a02727d9ff3ef6f8e672a39a19ce64a7, file:
#   voicemem/leftbrain/time_expand.py
# Changes: comments and docstrings translated to English. ENGLISH SUPPORT ADDED:
#   upstream recognised only Chinese expressions and emitted only the Chinese
#   date format, so the function was a no-op on English input. English triggers
#   now emit "March 14, 2025", which is the format the extraction prompt
#   normalises English dates to (verified against the worked examples in
#   extraction/data/additive_extraction_prompt.txt). Chinese behaviour, trigger
#   words, output format and the 8 day cap are unchanged.
# See CHANGES-FROM-UPSTREAM.md.
"""Expand relative time words in a query into absolute dates before retrieval.

Why this exists. Extraction normalises dates when it stores a memory: "health
check next Wednesday at 3pm" is stored as "... on August 26, 2026 at 3pm". The
stored text carries an absolute date. But the question the user later asks,
"what do I have next week", contains no absolute date at all, so the vectors
have nothing to match on. Upstream measured the difference:

    "what do I have next week"    0 of 3 scheduled items retrieved
    "what do I have on Aug 26"    3 of 3 retrieved

Same memories, same store, only the phrasing differs. So before searching, the
relative expression is expanded in place and the dates are appended to the query
text. Only the text used *for retrieval* is changed; the user's actual words are
untouched and nothing extra is written to memory.

    expand_relative_dates("what do I have next week")
    -> "what do I have next week (August 31, 2026 ... September 6, 2026)"

No relative expression means the query is returned unchanged. Pure regex, no
model, no network, no cost.

**The output format is load-bearing.** Appending a date in a format the stored
memories never use is worse than appending nothing, because it dilutes the
query's own semantics for no matching gain. The two formats here are each
matched to how the extractor writes that language:

* English -> ``March 14, 2025``. Verified against the extraction prompt's worked
  examples, which produce "around March 14, 2025", "in April 2025" and "the week
  of May 15, 2023".
* Chinese -> ``2026年8月26日``. Unchanged from upstream.

A query containing both languages gets both formats appended, since either may
be what the memory was stored under.

Weeks run Monday to Sunday in both languages, following upstream. A US reader
saying "next week" on a Sunday may mean something slightly different; the seven
day span still covers it.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Chinese (upstream)
# ---------------------------------------------------------------------------

#: Relative expression -> (start offset in days from today, number of days).
#: Week expressions compute their offset from the current weekday at resolve
#: time, so they carry ``None`` here as a placeholder.
_ZH_SPANS: dict[str, tuple[int | None, int]] = {
    "前天": (-2, 1),
    "昨天": (-1, 1),
    "今天": (0, 1),
    "今日": (0, 1),
    "明天": (1, 1),
    "明日": (1, 1),
    "后天": (2, 1),
    "大后天": (3, 1),
    "这几天": (0, 3),
    "最近几天": (-3, 4),
    "接下来几天": (0, 4),
    "未来几天": (0, 4),
    "上周": (None, 7),
    "上个星期": (None, 7),
    "这周": (None, 7),
    "本周": (None, 7),
    "这个星期": (None, 7),
    "下周": (None, 7),
    "下个星期": (None, 7),
    "下星期": (None, 7),
}

#: Chinese week expressions -> offset in weeks from the Monday of this week.
_ZH_WEEK_OFFSET = {
    "上周": -1,
    "上个星期": -1,
    "这周": 0,
    "本周": 0,
    "这个星期": 0,
    "下周": 1,
    "下个星期": 1,
    "下星期": 1,
}

# ---------------------------------------------------------------------------
# English
# ---------------------------------------------------------------------------

_EN_SPANS: dict[str, tuple[int | None, int]] = {
    "the day before yesterday": (-2, 1),
    "day before yesterday": (-2, 1),
    "yesterday": (-1, 1),
    "today": (0, 1),
    "tonight": (0, 1),
    "this morning": (0, 1),
    "this afternoon": (0, 1),
    "this evening": (0, 1),
    "tomorrow": (1, 1),
    "the day after tomorrow": (2, 1),
    "day after tomorrow": (2, 1),
    "the past few days": (-3, 4),
    "past few days": (-3, 4),
    "the last few days": (-3, 4),
    "last few days": (-3, 4),
    "the next few days": (0, 4),
    "next few days": (0, 4),
    "coming days": (0, 4),
    "last week": (None, 7),
    "the last week": (None, 7),
    "past week": (None, 7),
    "the past week": (None, 7),
    "this week": (None, 7),
    "next week": (None, 7),
    "the next week": (None, 7),
    "coming week": (None, 7),
    "the coming week": (None, 7),
}

_EN_WEEK_OFFSET = {
    "last week": -1,
    "the last week": -1,
    "past week": -1,
    "the past week": -1,
    "this week": 0,
    "next week": 1,
    "the next week": 1,
    "coming week": 1,
    "the coming week": 1,
}

#: Longest match first, so "the day after tomorrow" is not consumed by
#: "tomorrow" and "下个星期" is not shredded by "下星期".
_ZH_RE = re.compile("|".join(re.escape(w) for w in sorted(_ZH_SPANS, key=len, reverse=True)))
#: Word-bounded so "today" does not fire inside "todays" and "this week" does
#: not fire inside "this weekend", which means something else entirely.
_EN_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in sorted(_EN_SPANS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

#: Cap on how many dates one expansion may append, per language. "the last
#: three months" would expand to hundreds of dates, diluting the query's own
#: semantics enough that retrieval gets worse rather than better.
_MAX_DAYS = 8


def _resolve(word: str, today: date, spans: dict, week_offset: dict) -> list[date]:
    """Which days one relative expression covers."""
    if word in week_offset:
        monday = today - timedelta(days=today.weekday())
        start = monday + timedelta(weeks=week_offset[word])
        return [start + timedelta(days=i) for i in range(7)]
    offset, days = spans[word]
    start = today + timedelta(days=offset or 0)
    return [start + timedelta(days=i) for i in range(days)]


def _collect(words: list[str], today: date, spans: dict, week_offset: dict) -> list[date]:
    """Sorted, deduplicated days for a set of matches, or [] if over the cap."""
    days: list[date] = []
    for word in words:
        for d in _resolve(word, today, spans, week_offset):
            if d not in days:
                days.append(d)
    if not days or len(days) > _MAX_DAYS:
        return []
    days.sort()
    return days


def _fmt_en(d: date) -> str:
    """``March 14, 2025``. Matches how the extractor normalises English dates.

    Built by hand rather than with strftime("%B %-d, %Y"): the no-pad directive
    is platform specific and %B is locale dependent, and this string has to be
    byte-identical to what the prompt produces regardless of where it runs.
    """
    months = (
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    )
    return f"{months[d.month - 1]} {d.day}, {d.year}"


def _fmt_zh(d: date) -> str:
    """``2026年8月26日``. Unchanged from upstream."""
    return f"{d.year}年{d.month}月{d.day}日"


def expand_relative_dates(query: str, today: date | None = None) -> str:
    """Append the absolute dates a query refers to.

    Returns the query unchanged when it contains no recognised relative
    expression in either language, or when the expansion would exceed the cap.
    """
    if not query:
        return query

    en_words = [w.lower() for w in _EN_RE.findall(query)]
    zh_words = _ZH_RE.findall(query)
    if not en_words and not zh_words:
        return query

    today = today or date.today()
    out = query

    if en_words:
        days = _collect(en_words, today, _EN_SPANS, _EN_WEEK_OFFSET)
        if days:
            out = f"{out} ({' '.join(_fmt_en(d) for d in days)})"
    if zh_words:
        days = _collect(zh_words, today, _ZH_SPANS, _ZH_WEEK_OFFSET)
        if days:
            out = f"{out}（{' '.join(_fmt_zh(d) for d in days)}）"

    return out
