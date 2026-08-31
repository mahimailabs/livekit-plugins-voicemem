#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Assert that a git tag matches the packaged version.

A mismatched tag is the most common release-day mistake and the most annoying
one, because a version number can never be reused on PyPI. One comparison
prevents it.

    python scripts/check_version_tag.py v0.1.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parent.parent / "livekit/plugins/voicemem/version.py"


def packaged_version() -> str:
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
    if not match:
        sys.exit(f"could not find __version__ in {VERSION_FILE}")
    return match.group(1)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.exit("usage: check_version_tag.py <tag>")
    tag = argv[0]
    if not tag.startswith("v"):
        sys.exit(f"tag {tag!r} must start with 'v'")

    wanted, actual = tag[1:], packaged_version()
    if wanted != actual:
        sys.exit(
            f"tag {tag!r} does not match the packaged version {actual!r}.\n"
            f"Either retag, or set __version__ = \"{wanted}\" in {VERSION_FILE.name}."
        )

    if ".dev" in actual or actual.endswith(("a0", "b0")):
        # A dev version reaching a release tag means the bump was forgotten.
        sys.exit(f"refusing to release a development version: {actual!r}")

    changelog = VERSION_FILE.parent.parent.parent.parent / "CHANGELOG.md"
    if changelog.exists() and f"[{actual}]" not in changelog.read_text(encoding="utf-8"):
        sys.exit(f"CHANGELOG.md has no section for [{actual}]. Write it before releasing.")

    print(f"tag {tag} matches version {actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
