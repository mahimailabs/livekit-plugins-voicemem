#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mahimai Labs
#
# Cut a release. Everything after the tag push is automated by publish.yml.
#
#     scripts/release.sh 0.1.0
#     git push origin main --follow-tags
#
# Deliberately does not push. Read the diff, then push.
set -euo pipefail

VERSION="${1:-}"
[ -n "$VERSION" ] || { echo "usage: scripts/release.sh <version>   e.g. 0.1.0" >&2; exit 2; }
VERSION="${VERSION#v}"

cd "$(dirname "$0")/.."
VERSION_FILE="livekit/plugins/voicemem/version.py"

[ -z "$(git status --porcelain)" ] || { echo "working tree is dirty; commit or stash first" >&2; exit 1; }

echo "==> setting version to $VERSION"
# uv run, not bare `python`: that is not on PATH on macOS.
uv run python - "$VERSION" <<'PY'
import re, sys
from pathlib import Path
version = sys.argv[1]
p = Path("livekit/plugins/voicemem/version.py")
p.write_text(re.sub(r'__version__\s*=\s*["\'][^"\']+["\']',
                    f'__version__ = "{version}"', p.read_text()))
PY

echo "==> checks"
# Build into a clean dist so the twine glob below cannot match a stale
# pre-release sibling (dist/*0.1.0* also matches 0.1.0.dev0).
rm -rf dist
uv run ruff check .
uv run python scripts/check_version_tag.py "v$VERSION"
uv build >/dev/null
uv run --with twine twine check --strict dist/*"$VERSION"* 

echo "==> committing and tagging"
git add "$VERSION_FILE" CHANGELOG.md
# Nothing to commit when version.py already holds the target version, which is
# the normal case when the bump was made by hand. Tag regardless: without this
# the script prints a wall of green and then dies here without tagging, and the
# subsequent push silently releases nothing.
git diff --cached --quiet || git commit -m "Release $VERSION"
git tag -a "v$VERSION" -m "v$VERSION"

cat <<EOF

Ready. Nothing has been pushed yet.

  git push origin main --follow-tags

That triggers publish.yml: verify, build, then publish to PyPI once the
'pypi' environment is approved.
EOF
