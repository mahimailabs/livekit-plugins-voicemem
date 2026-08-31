#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
"""Every source file carries a licence header, and derived files say so.

Run as its own CI job from the first vendoring commit rather than as a sweep at
the end. A missing Apache-2.0 header on a derived file is exactly what an
automated licence scanner at a downstream user will flag, and finding them all
after the fact is much harder than never losing one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANGES = ROOT / "CHANGES-FROM-UPSTREAM.md"

SPDX = "SPDX-License-Identifier: Apache-2.0"
DERIVED_MARKER = "Derived from VoiceMem"


def main() -> int:
    problems: list[str] = []
    derived: list[Path] = []

    for path in sorted(ROOT.glob("livekit/**/*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()[:20]
        rel = path.relative_to(ROOT)
        if not any(SPDX in ln for ln in lines):
            problems.append(f"{rel}: missing '{SPDX}' in the first 20 lines")

        # Provenance lives in the COMMENT header, not in the docstring. A module
        # that merely mentions upstream in its prose is not itself derived, and
        # demanding a changes note from it would train people to add noise.
        comments = "\n".join(ln for ln in lines if ln.lstrip().startswith("#"))
        if any(m in comments for m in (DERIVED_MARKER, "Replaces VoiceMem", "Replaces the SQLite")):
            derived.append(rel)
            if "Changes:" not in comments:
                problems.append(
                    f"{rel}: derived from upstream but its header states no changes "
                    "(Apache-2.0 section 4(b))"
                )

    for path in sorted(ROOT.glob("scripts/*.py")):
        head = "\n".join(path.read_text(encoding="utf-8").splitlines()[:10])
        if SPDX not in head:
            problems.append(f"{path.relative_to(ROOT)}: missing '{SPDX}'")

    if not CHANGES.exists():
        problems.append("CHANGES-FROM-UPSTREAM.md is missing; Apache-2.0 4(b) requires it")

    if problems:
        print("licence header check failed:\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    print(f"licence headers OK ({len(derived)} file(s) derived from upstream)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
