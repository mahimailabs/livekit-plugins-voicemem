# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
from livekit.plugins.voicemem.instrument import LLMCall, Recorder

CALLS = ("extract", "conflict", "slot_tag", "inner_os", "attribution")


def test_records_the_five_calls_a_turn_costs():
    r = Recorder()
    r.begin_turn()
    for p in CALLS:
        r.record_llm(LLMCall(purpose=p, model="gpt-4o-mini", duration_s=0.1))
    t = r.end_turn()
    assert t.llm_call_count == 5
    assert round(t.llm_total_s, 2) == 0.5
    assert r.counters["llm.extract"] == 1


def test_stages_accumulate_and_are_reported_separately():
    r = Recorder()
    r.begin_turn()
    with r.stage("rank"):
        pass
    with r.stage("rank"):
        pass
    with r.stage("rb"):
        pass
    t = r.end_turn()
    assert set(t.recall_ms) == {"rank", "rb"}
    assert t.recall_ms["rank"] >= 0.0


def test_stage_timing_survives_an_exception():
    r = Recorder()
    r.begin_turn()
    try:
        with r.stage("rank"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "rank" in r.end_turn().recall_ms


def test_prefetch_hit_rate():
    r = Recorder()
    for hit in (True, False, True, True):
        r.begin_turn()
        r.record_recall(total_ms=10.0, hits=2, rb_hits=1, prefetched=hit)
        r.end_turn()
    assert r.prefetch_hit_rate == 0.75
    assert r.summary()["turns"] == 4


def test_hit_rate_is_zero_not_a_crash_before_any_turn():
    assert Recorder().prefetch_hit_rate == 0.0


def test_budget_exhaustion_is_counted_because_it_is_invisible_otherwise():
    r = Recorder()
    r.begin_turn()
    r.record_budget_exhausted()
    r.end_turn()
    assert r.summary()["budget_exhausted"] == 1


def test_errors_are_counted_separately():
    r = Recorder()
    r.begin_turn()
    r.record_llm(LLMCall(purpose="extract", model="m", duration_s=0.1, ok=False))
    r.end_turn()
    assert r.counters["llm.extract.error"] == 1


def test_a_broken_metrics_hook_cannot_break_the_memory_path():
    def boom(event, payload):
        raise RuntimeError("sink is down")

    r = Recorder(metrics_hook=boom)
    r.begin_turn()
    r.end_turn()  # must not raise


def test_two_recorders_do_not_share_state():
    a, b = Recorder(), Recorder()
    a.begin_turn()
    a.record_llm(LLMCall(purpose="extract", model="m", duration_s=0.1))
    a.end_turn()
    assert b.counters == {}
