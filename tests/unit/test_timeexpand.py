# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Pure-tier tests: no database, no network, no mocks, nothing injected."""

from datetime import date

import pytest

from livekit.plugins.voicemem.leftbrain.timeexpand import expand_relative_dates

# A Wednesday, so weekday arithmetic is exercised away from the boundaries.
TODAY = date(2026, 8, 26)


def test_no_relative_expression_returns_query_unchanged():
    assert expand_relative_dates("我对什么过敏", TODAY) == "我对什么过敏"


def test_empty_query_is_returned_as_is():
    assert expand_relative_dates("", TODAY) == ""


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("昨天", "2026年8月25日"),
        ("今天", "2026年8月26日"),
        ("明天", "2026年8月27日"),
        ("后天", "2026年8月28日"),
        ("前天", "2026年8月24日"),
    ],
)
def test_single_day_expressions(word, expected):
    out = expand_relative_dates(f"我{word}干了什么", TODAY)
    assert out == f"我{word}干了什么（{expected}）"


def test_next_week_expands_to_that_whole_week():
    # Monday of the week containing 2026-08-26 is 2026-08-24, so "next week"
    # is 2026-08-31 through 2026-09-06.
    out = expand_relative_dates("我下周有什么安排", TODAY)
    assert "2026年8月31日" in out
    assert "2026年9月6日" in out
    assert out.count("年") == 7


def test_dates_are_sorted_and_deduplicated():
    # "today" is inside "this week", so the union must not repeat it.
    out = expand_relative_dates("今天和这周", TODAY)
    body = out.split("（")[1].rstrip("）")
    stamps = body.split()
    assert stamps == sorted(set(stamps), key=stamps.index)
    assert len(stamps) == len(set(stamps))


def test_expansion_is_capped_and_falls_back_to_the_raw_query():
    # Two different weeks is 14 days, over the 8 day cap, so nothing is appended.
    out = expand_relative_dates("上周和下周", TODAY)
    assert out == "上周和下周"


def test_longest_trigger_wins():
    out = expand_relative_dates("下个星期", TODAY)
    assert "2026年8月31日" in out


# --- English -------------------------------------------------------------
# The emitted format must match what the extraction prompt writes for English
# ("around March 14, 2025" in its worked examples). A format the stored
# memories never use is worse than appending nothing.


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("what did I do yesterday", "August 25, 2026"),
        ("my plans tomorrow", "August 27, 2026"),
        ("anything today", "August 26, 2026"),
        ("the day after tomorrow", "August 28, 2026"),
        ("the day before yesterday", "August 24, 2026"),
    ],
)
def test_english_single_day(q, expected):
    assert expand_relative_dates(q, TODAY) == f"{q} ({expected})"


def test_english_next_week_spans_that_week():
    out = expand_relative_dates("what do I have next week", TODAY)
    assert "August 31, 2026" in out and "September 6, 2026" in out
    assert out.count(",") == 7


def test_english_is_case_insensitive():
    assert "August 25, 2026" in expand_relative_dates("What did I do YESTERDAY", TODAY)


def test_longest_english_trigger_wins_over_substring():
    # "the day after tomorrow" must not be consumed by "tomorrow".
    out = expand_relative_dates("the day after tomorrow", TODAY)
    assert out.endswith("(August 28, 2026)")


def test_english_respects_word_boundaries():
    # "this weekend" is not "this week"; firing here would append a whole week
    # of wrong dates to an unrelated question.
    assert expand_relative_dates("plans for this weekend", TODAY) == "plans for this weekend"


def test_english_over_cap_falls_back_to_raw_query():
    assert expand_relative_dates("last week and next week", TODAY) == "last week and next week"


def test_date_format_matches_the_extraction_prompt():
    # Guards the pairing this whole feature depends on. No zero padding, full
    # month name, comma before the year.
    out = expand_relative_dates("today", date(2026, 3, 4))
    assert out == "today (March 4, 2026)"


def test_mixed_language_query_gets_both_formats():
    out = expand_relative_dates("what about 明天 and tomorrow", TODAY)
    assert "August 27, 2026" in out
    assert "2026年8月27日" in out
