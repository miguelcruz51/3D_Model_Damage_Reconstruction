"""Unit tests for README/docs completeness and .gitignore patterns.

Covers the documentation and hygiene acceptance criteria:
- README contains the required sections (Req 9.1-9.6).
- README contains the train/eval/inference command strings (Req 9.4).
- .gitignore contains the required ignore patterns (Req 10.4).
"""

import os

import pytest

_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel_path):
    with open(os.path.join(_CLEAN_REPO_ROOT, rel_path), encoding="utf-8") as handle:
        return handle.read()


# Required README section headings (Req 9.1, 9.2, 9.3, 9.5, 9.6).
README_SECTIONS = [
    "## Overview",
    "## Problem",
    "## Reconstruction Pipeline",
    "## Method",
    "## Dataset",
    "## Installation",
    "## Usage",
    "## Results",
    "## Repository Structure",
    "## Research Context",
    "## Files to Inspect First",
    "## Citation",
    "## License",
]

# Required command substrings (Req 9.4).
README_COMMANDS = [
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/inference.py",
]

# Required .gitignore patterns (Req 10.4): caches, editor files, virtualenvs,
# large datasets, checkpoints.
GITIGNORE_PATTERNS = [
    "__pycache__/",
    ".vscode/",
    ".idea/",
    "venv/",
    "env/",
    "data/",
    "*.pth",
    "log/",
]


@pytest.mark.parametrize("section", README_SECTIONS)
def test_readme_contains_required_section(section):
    """README declares every required section heading."""
    readme = _read("README.md")
    assert section in readme, f"README missing required section: {section!r}"


@pytest.mark.parametrize("command", README_COMMANDS)
def test_readme_contains_command(command):
    """README documents the train/eval/inference commands."""
    readme = _read("README.md")
    assert command in readme, f"README missing command reference: {command!r}"


@pytest.mark.parametrize("pattern", GITIGNORE_PATTERNS)
def test_gitignore_contains_pattern(pattern):
    """.gitignore excludes caches, editor files, venvs, datasets, checkpoints."""
    gitignore = _read(".gitignore")
    assert pattern in gitignore, f".gitignore missing pattern: {pattern!r}"
