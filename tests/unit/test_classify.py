# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Slot selection: an absolute floor, a relative window, or both.

The floor is what OpenAI embeddings were calibrated with. The window exists
because E5's cosines are bunched so tightly that no floor separates a right slot
from a wrong one, so at 0.72 every slot passes for every memory and the filter
silently stops filtering. These pin both rules so neither can drift into the
other's backend.
"""

from __future__ import annotations

from livekit.plugins.voicemem.leftbrain.classify import SlotClassifier

SCORED = [(0.90, "work"), (0.89, "goals"), (0.80, "health"), (0.60, "finance")]


def _classifier(**kw) -> SlotClassifier:
    return SlotClassifier(embedder=None, **kw)  # type: ignore[arg-type]


def test_absolute_floor_keeps_what_clears_it() -> None:
    picked = _classifier(min_score=0.85, max_slots=4)._select(SCORED)
    assert [name for _s, name in picked] == ["work", "goals"]


def test_a_floor_that_everything_clears_selects_everything() -> None:
    """This is the E5 failure, reproduced. 0.72 against bunched scores is not a
    filter, it is a pass-through, and nothing in the old code said so."""
    picked = _classifier(min_score=0.5, max_slots=4)._select(SCORED)
    assert len(picked) == 4


def test_relative_window_keeps_only_what_is_close_to_the_best() -> None:
    picked = _classifier(min_score=0.0, margin=0.02, max_slots=4)._select(SCORED)
    assert [name for _s, name in picked] == ["work", "goals"]


def test_a_narrow_window_keeps_the_single_best() -> None:
    picked = _classifier(min_score=0.0, margin=0.005, max_slots=4)._select(SCORED)
    assert [name for _s, name in picked] == ["work"]


def test_window_is_relative_so_it_survives_a_shifted_scale() -> None:
    """The point of the window: it does not care where the scores sit, only how
    far apart they are. An absolute floor cannot make that claim."""
    shifted = [(s - 0.15, n) for s, n in SCORED]
    assert _classifier(min_score=0.0, margin=0.02, max_slots=4)._select(
        shifted
    ) == [(0.75, "work"), (0.74, "goals")]


def test_both_rules_apply_together() -> None:
    picked = _classifier(min_score=0.85, margin=0.5, max_slots=4)._select(SCORED)
    assert [name for _s, name in picked] == ["work", "goals"]


def test_max_slots_caps_the_result() -> None:
    picked = _classifier(min_score=0.0, margin=1.0, max_slots=2)._select(SCORED)
    assert len(picked) == 2


def test_zero_slots_disables_narrowing() -> None:
    """How the local backend turns slot filtering off: no slots means recall
    searches the whole corpus rather than a guessed subset."""
    assert _classifier(min_score=0.0, margin=1.0, max_slots=0)._select(SCORED) == []


def test_no_scores_selects_nothing() -> None:
    """An empty result means "search everything", which is slow but correct. A
    wrong slot silently hides the right memory, which is neither."""
    assert _classifier(min_score=0.0, margin=0.01)._select([]) == []


def test_default_is_the_absolute_floor_and_no_window() -> None:
    """Constructed bare, behaviour must be exactly what OpenAI had before."""
    default = _classifier()
    assert default._margin is None
    assert default._min_score == 0.72
    assert default._max_slots == 2


# -- the rule has to actually reach the callers ----------------------------


async def test_classify_applies_the_window(monkeypatch) -> None:
    """_select is only worth having if classify() goes through it."""

    async def fake_scored(self, text, vector):
        return SCORED

    monkeypatch.setattr(SlotClassifier, "_scored", fake_scored)
    result = await _classifier(min_score=0.0, margin=0.02, max_slots=4).classify("anything")
    assert result.slots == ("work", "goals")


async def test_tag_applies_the_window(monkeypatch) -> None:
    """tag() writes to memory_tags, so a rule applied to one and not the other
    would store slots that retrieval then refuses to match."""

    async def fake_scored(self, text, vector):
        return SCORED

    monkeypatch.setattr(SlotClassifier, "_scored", fake_scored)
    tags = await _classifier(min_score=0.0, margin=0.02, max_slots=4).tag("anything")
    assert [name for name, _score in tags] == ["work", "goals"]


# -- per-backend tuning ----------------------------------------------------


def test_the_container_gives_each_backend_its_own_rule() -> None:
    """OpenAI keeps the absolute floor it was calibrated with; local gets the
    relative window, because its cosines are too bunched for any floor."""
    from livekit.plugins.voicemem.config import Config
    from livekit.plugins.voicemem.container import _slot_classifier

    cfg = Config(pg_dsn="postgresql://x/y", openai_api_key="k")

    openai = _slot_classifier(cfg, None, "openai")  # type: ignore[arg-type]
    assert (openai._min_score, openai._margin, openai._max_slots) == (0.72, None, 2)

    local = _slot_classifier(cfg, None, "local")  # type: ignore[arg-type]
    assert local._max_slots == 0, (
        "E5 cannot tell slot-bearing text from vague text by any measured "
        "statistic, and a partly-wrong guess hard-excludes the right memory, "
        "so local searches the whole corpus instead of narrowing"
    )


def test_explicit_config_overrides_the_backend_defaults() -> None:
    from livekit.plugins.voicemem.config import Config
    from livekit.plugins.voicemem.container import _slot_classifier

    cfg = Config(
        pg_dsn="postgresql://x/y",
        openai_api_key="k",
        slot_min_score=0.5,
        slot_margin=0.2,
        slot_max_slots=7,
    )
    tuned = _slot_classifier(cfg, None, "local")  # type: ignore[arg-type]
    assert (tuned._min_score, tuned._margin, tuned._max_slots) == (0.5, 0.2, 7)
