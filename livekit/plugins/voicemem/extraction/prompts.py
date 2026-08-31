# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Derived from VoiceMem (https://github.com/xzf-thu/VoiceMem), Apache-2.0,
# commit d587d424a02727d9ff3ef6f8e672a39a19ce64a7, files:
#   voicemem/leftbrain/merged_extraction.py       (PROMPT_ADDENDUM, lines 86-149)
#   voicemem/leftbrain/mem0_additive_prompt_build.py  (user-message assembly)
# Changes: the merge addendum is REWRITTEN IN ENGLISH. Upstream's version is
#   written in Chinese and instructs the model to emit Chinese trait labels and a
#   single Chinese emotion word, which is wrong for an English-first plugin: the
#   labels become node titles the user reads, and the emotion becomes the anchor
#   every right-brain note hangs from. Structure, JSON shape and the emotion-vs-
#   coping distinction are preserved; the five trait slots are renamed from
#   Chinese to English identifiers.
#   The two .txt prompts under data/ are byte-identical to upstream, which took
#   them from mem0 (Apache-2.0) unchanged. See NOTICE.
# See CHANGES-FROM-UPSTREAM.md.
"""Prompt assembly. Pure string work: no I/O, no async, no model calls."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from importlib import resources
from typing import Any

from ..leftbrain.slots import ALL_SLOT_V2_VALUES

__all__ = [
    "TRAIT_SLOTS",
    "build_conflict_user_message",
    "build_extraction_user_message",
    "conflict_system_prompt",
    "extraction_system_prompt",
]

#: The five kinds of judgement the right brain stores. Renamed from upstream's
#: Chinese names. These are written to rb_traits.slot, so changing them later
#: requires a data migration.
TRAIT_SLOTS = ("emotion", "coping", "expression", "thinking", "preferences")


@lru_cache(maxsize=2)
def _load(name: str) -> str:
    """Read a prompt shipped as package data.

    Through importlib.resources so it works from a wheel. Cached because these
    are 34KB and read on every extraction otherwise. Upstream read one of them
    at import time, which turns a missing file into an ImportError.
    """
    text = resources.files(__package__).joinpath("data", name).read_text(encoding="utf-8")
    if not text.strip():
        raise RuntimeError(f"prompt file {name} is present but empty")
    return text


#: Folds three LLM calls into one. Upstream measured a single ingest issuing
#: twelve chat completions; merging annotation and trait extraction into the
#: extraction call removes two of them. When the model fails to honour the
#: shape, each consumer falls back to its own call, so the failure mode is cost
#: rather than breakage.
_ADDENDUM = """

Additionally, for EACH item in "memory", include these three fields:
- "slot": one of [{slots}]
- "entities": [{{"name": "...", "entity_type": "user|person|project|task|knowledge|preference|place|routine|asset|organization|event", "role": "subject|object|context|owner"}}]
- "relations": [{{"from": "...", "to": "...", "relation_type": "...", "confidence": 0.9}}]
  Relation direction must match reality: a manager manages the user, not the reverse.

And add ONE top-level field "traits": subjective things this utterance reveals
about the speaker. Each item {{"slot": "...", "label": "..."}}, where slot is one of:

  emotion      WHEN they feel WHAT: the situation together with the feeling it
               triggers. "gets tense before reviews", "irritated by interruptions"
  coping       what they DO about a feeling, or how they want to be treated.
               "wants reassurance under pressure", "needs space when upset"
  expression   habits of speaking and communicating
  thinking     how they reason, weigh options, decide
  preferences  what they like or dislike

emotion versus coping is the distinction people get wrong: "irritated by
interruptions" is emotion (a feeling appearing), "walks away when interrupted"
is coping (an action taken). If the label contains no verb of doing or wanting,
it is emotion.

An emotion label must read as A PATTERN, NOT A BARE FEELING WORD:
"gets tense before reviews", "calmer working alone" and NOT "anxious" or "happy".
It becomes the title of a node the user can read; a bare word tells them nothing.

When the utterance states a RECURRING tendency about the speaker: "whenever I",
"every time", "always", "never", "I'm the kind of person who", or any habit or
reaction that clearly holds beyond this one moment, a trait is REQUIRED.
"I zone out in long meetings" is preferences: "dislikes long meetings".
"I never sleep before a presentation" is emotion: "sleepless before presentations".
Do not skip it because the same content also went into "memory". "memory"
records WHAT HAPPENED; "traits" records WHAT THIS PERSON IS LIKE. One sentence
very often carries both.

Outside that case, include only a category the utterance clearly shows. For a
one-off event or a plain question, "traits": [] is the right answer.

Each label becomes the TITLE of a node, so write it as a short pattern of three
to eight words, no subject, no full stop:
  good: dislikes being interrupted / wants the conclusion before the reasoning
  bad:  The user tends to plan in detail and think in a structured way.
        (a full sentence with a subject)
  bad:  studies computer science  (copying the utterance as a fact)

Also add ONE top-level field "emotion": how the speaker feels, as a single
lowercase English word (happy, calm, anxious, sad, hurt, angry, surprised,
tired, disappointed, frustrated, excited, ...).
**Judge from what they actually say.** "I love strawberries" is happy, not sad.
"I'm so angry" is angry, not anxious. If the utterance carries no clear feeling,
such as a plain fact or a question, return "" — an empty string is the right
answer far more often than a guess. A wrong label is worse than none: it is
shown next to the user's own words and it anchors what gets recalled later.

Never invent entities, traits or feelings that are not in the text.

Keep one-off requests OUT of "memory": asking for a recommendation, asking what
you remember, asking you to do something right now. When the same sentence ALSO
states a lasting fact, write only the lasting half, never both in one item.
"I'm taking the GRE next week, got any book recommendations?" gives exactly one
memory, "User is taking the GRE next week", and nothing about the book request.

OUTPUT SHAPE: your JSON object must have EXACTLY these three top-level keys:
{{"memory": [...], "emotion": "...", "traits": [...]}}
The prompt above describes only the "memory" key. "emotion" and "traits" are
REQUIRED as well; omitting them is an error. Use "" and [] when there is nothing.
"""


def extraction_system_prompt(*, merged: bool = True) -> str:
    """The extraction system prompt, optionally with the merge addendum."""
    base = _load("additive_extraction_prompt.txt")
    if not merged:
        return base
    return base + _ADDENDUM.format(slots=", ".join(ALL_SLOT_V2_VALUES))


def conflict_system_prompt() -> str:
    return _load("mem0_update_memory_prompt.txt")


def build_extraction_user_message(
    *,
    user_text: str,
    agent_reply: str = "",
    existing: list[dict[str, str]] | None = None,
    observation_date: str | None = None,
    current_date: str | None = None,
) -> str:
    """Assemble the user-side message.

    The two dates are distinct and the distinction matters. Observation Date
    grounds relative expressions in the utterance; Current Date may be years
    later. Upstream's prompt is explicit that resolving "recently" against the
    current date rather than the observation date is wrong.
    """
    today = date.today().isoformat()
    obs = observation_date or today
    cur = current_date or today

    messages: list[dict[str, str]] = [{"role": "user", "content": user_text}]
    if agent_reply:
        # Included so the extractor can disambiguate a reply like "yes, that one"
        # which is meaningless without what it answered.
        messages.append({"role": "assistant", "content": agent_reply})

    parts = [
        "Summary: \"\"",
        "Recently Extracted: []",
        f"Existing Memories: {json.dumps(existing or [], ensure_ascii=False)}",
        "New Messages:",
        json.dumps(messages, ensure_ascii=False),
        f"Observation Date: {obs}",
        f"Current Date: {cur}",
        "",
        "Output:",
    ]
    return "\n".join(parts)


def build_conflict_user_message(
    *, existing: list[dict[str, Any]], new_facts: list[str], speaker: str | None = None
) -> str:
    """Assemble the ADD/UPDATE/DELETE/NONE decision as ONE user message.

    The whole thing goes in a single user message, system prompt included,
    rather than being split across roles. That is not a style choice, and
    upstream documents the reason: mem0's conflict prompt does not contain the
    word "JSON" anywhere, and OpenAI rejects ``response_format=json_object``
    with a 400 unless some message does. The output-structure block below
    supplies it, and it doubles as the field-by-field contract the parser reads.

    Existing memories are presented with integer indices rather than UUIDs, also
    deliberately: models produce plausible-looking UUIDs out of thin air, while
    an invented index is out of range and therefore detectable.
    """
    numbered = [{"id": str(i), "text": m.get("text", "")} for i, m in enumerate(existing)]
    memory_part = (
        "Below is the current content of my memory which I have collected till now. "
        "You have to update it in the following format only:\n\n"
        f"```\n{json.dumps(numbered, ensure_ascii=False, indent=2)}\n```\n"
        if existing
        else "Current memory is empty.\n"
    )
    speaker_part = (
        f"These new facts were all spoken by: {speaker}. Use this as the authoritative "
        "subject when checking whether an existing memory is about the same person.\n"
        if speaker
        else ""
    )

    return f"""{conflict_system_prompt()}

{memory_part}
The new retrieved facts are mentioned in the triple backticks. You have to analyze
the new retrieved facts and determine whether these facts should be added, updated,
or deleted in the memory.
{speaker_part}
```
{json.dumps(new_facts, ensure_ascii=False, indent=2)}
```

You must return your response in the following JSON structure only:

{{
    "memory": [
        {{
            "id": "<ID of the memory>",
            "text": "<Content of the memory>",
            "event": "<Operation to be performed>",
            "old_memory": "<Old memory content, only when event is UPDATE>"
        }}
    ]
}}

Follow the instructions below:
- Do not return anything from the few-shot examples above.
- If the current memory is empty, add the new retrieved facts to the memory.
- The id for UPDATE and DELETE must be one of the ids shown above, unchanged.
- Return only the JSON object, with no surrounding prose.
"""
