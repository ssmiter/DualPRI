#!/usr/bin/env python3
"""Run dependency-free checks on the iSCALE release tree."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "README.md",
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "environment.yml",
    "requirements.txt",
    "configs/paper_config.yaml",
    "docs/DATA.md",
    "docs/REPRODUCIBILITY.md",
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".cff", ".json"}
BANNED_PATTERNS = {
    "internal server path": re.compile(r"/media/[A-Za-z0-9_.-]+/"),
    "local wheel URL": re.compile(r"file:///"),
    "conda environment prefix": re.compile(r"^prefix:\s", re.MULTILINE),
    "private key": re.compile(r"BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY"),
    "non-English CJK text": re.compile(r"[\u4e00-\u9fff]"),
}


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"Missing required file: {relative}")

    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            errors.append(f"Tracked cache candidate: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            warnings.append(f"Large release file (>10 MiB): {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                warnings.append(f"Non-UTF-8 text candidate: {relative}")
                continue
            if relative != Path("scripts/check_release.py"):
                for label, pattern in BANNED_PATTERNS.items():
                    if pattern.search(text):
                        errors.append(f"{label} found in {relative}")
            if path.suffix == ".py":
                try:
                    ast.parse(text, filename=str(relative))
                except SyntaxError as exc:
                    errors.append(f"Python syntax error in {relative}: {exc}")

    print(f"Repository: {ROOT}")
    print(f"Errors: {len(errors)}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        return 1
    print("Release tree checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
