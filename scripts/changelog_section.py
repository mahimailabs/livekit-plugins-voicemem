#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Print one version's CHANGELOG section, for the GitHub release notes."""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        sys.exit("usage: changelog_section.py <version>")
    version = argv[0].lstrip("v")
    text = CHANGELOG.read_text(encoding="utf-8")

    pattern = rf"^## \[{re.escape(version)}\].*?$(.*?)(?=^## |\Z)"
    match = re.search(pattern, text, re.M | re.S)
    if not match:
        sys.exit(f"no section for [{version}] in CHANGELOG.md")
    print(match.group(1).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
