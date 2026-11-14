#!/usr/bin/env python3
"""Documentation gate used locally and in CI.

  python scripts/check_docs.py

Checks, over every tracked Markdown file:
  1. relative links and images resolve to existing files (anchors on existing files are not validated)
  2. no stale project-framing or former-contributor wording in user-facing documents
  3. required governance files exist
Exit code 1 lists every problem. Files in ALLOW (historical audits that must quote the old wording) are skipped for check 2.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOW_WORDING = {"docs/development/GITHUB_REPOSITORY_AUDIT.md"}
BANNED = re.compile(r"\b(final[- ]year|b\.?tech|e&tc|college project|student project|capstone|team_responsibilities|submission freeze|pawan|gaurav)\b", re.I)
REQUIRED = ["README.md", "LICENSE", "CITATION.cff", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "CHANGELOG.md",
            ".env.example", ".github/workflows/ci.yml", ".github/pull_request_template.md"]
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def tracked_md() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files", "*.md"], cwd=ROOT, text=True).split()
    return [ROOT / p for p in out if not p.startswith(("results/", ".venv/"))]


def main() -> int:
    problems: list[str] = []
    for f in tracked_md():
        rel = f.relative_to(ROOT).as_posix()
        text = f.read_text(errors="replace")
        in_code = False
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            for m in LINK.finditer(line):
                t = m.group(1)
                if re.match(r"^(https?:|mailto:|#)", t):
                    continue
                path = t.split("#")[0]
                if path and not (f.parent / path).exists():
                    problems.append(f"{rel}:{n}: broken link -> {t}")
            if rel not in ALLOW_WORDING and BANNED.search(line):
                problems.append(f"{rel}:{n}: stale wording: {BANNED.search(line).group(0)!r}")
    for r in REQUIRED:
        if not (ROOT / r).exists():
            problems.append(f"missing required file: {r}")
    print("\n".join(problems) if problems else "docs check passed")
    print(f"({len(tracked_md())} markdown files checked, {len(problems)} problem(s))")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
