"""Pytest configuration for clean_repo tests.

Adds both the clean_repo root (for ``src.models``) and the original source
repository root (for the reference ``models`` package) to ``sys.path`` so that
behavior-preservation property tests can compare the two implementations.
"""

import os
import sys

# clean_repo/ (parent of this tests/ directory)
CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# original source repository root (parent of clean_repo/)
SOURCE_REPO_ROOT = os.path.dirname(CLEAN_REPO_ROOT)

for path in (CLEAN_REPO_ROOT, SOURCE_REPO_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)
