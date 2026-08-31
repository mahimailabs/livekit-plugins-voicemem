# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Long-term memory for LiveKit voice agents, backed by PostgreSQL and pgvector.

    from livekit.plugins import voicemem

    runtime = await voicemem.build(voicemem.Config(pg_dsn=..., openai_api_key=...))
    memory = runtime.session(user_id="alice")
    hooks = voicemem.MemoryHooks(memory)
    hooks.attach(session)

Derived from VoiceMem (https://github.com/xzf-thu/VoiceMem, Apache-2.0).
See NOTICE and CHANGES-FROM-UPSTREAM.md.
"""

from livekit.agents import Plugin

from .config import Config, PrefetchConfig, WriterConfig
from .container import Runtime, build
from .hooks import MemoryAgent, MemoryHooks, inject_memory
from .instrument import Recorder
from .log import logger
from .memory import VoiceMemory, render_context
from .types import MemoryHit, RecallResult, RightBrainHit, Scope, TurnRecord
from .version import __version__


class VoiceMemPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


# Registration is by import side effect; there are no entry points. This must
# stay at module scope: Plugin.register_plugin raises RuntimeError when called
# off the main thread.
Plugin.register_plugin(VoiceMemPlugin())

__all__ = [
    "Config",
    "MemoryAgent",
    "MemoryHit",
    "MemoryHooks",
    "PrefetchConfig",
    "RecallResult",
    "Recorder",
    "RightBrainHit",
    "Runtime",
    "Scope",
    "TurnRecord",
    "VoiceMemory",
    "WriterConfig",
    "__version__",
    "build",
    "inject_memory",
    "render_context",
]
