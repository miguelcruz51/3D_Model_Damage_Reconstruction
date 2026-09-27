"""Property-based test for absence of absolute machine paths (Property 7).

Scans every clean-repo source file under ``src/`` and ``scripts/`` and asserts
that none contains an absolute home-directory path (for example a path beginning
with ``/home/<user>/``, ``/Users/<user>/``, or a Windows ``C:\\Users\\<user>``).
"""

import os
import re

from hypothesis import given, settings
from hypothesis import strategies as st

_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOURCE_DIRS = ("src", "scripts")

# Absolute home-directory path patterns for the common platforms.
_HOME_PATH_PATTERNS = [
    re.compile(r"/home/[^/\s\"']+"),          # Linux
    re.compile(r"/Users/[^/\s\"']+"),         # macOS
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),    # Windows (escaped backslashes)
    re.compile(r"[A-Za-z]:\\Users\\"),        # Windows (literal backslashes)
]


def _iter_source_files():
    """Yield absolute paths of every ``.py`` file under the source dirs."""
    files = []
    for source_dir in _SOURCE_DIRS:
        root = os.path.join(_CLEAN_REPO_ROOT, source_dir)
        for dirpath, _dirnames, filenames in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            for name in filenames:
                if name.endswith(".py"):
                    files.append(os.path.join(dirpath, name))
    return files


_SOURCE_FILES = sorted(_iter_source_files())


@settings(max_examples=100, deadline=None)
@given(path=st.sampled_from(_SOURCE_FILES))
def test_no_absolute_machine_paths(path):
    """**Feature: repo-reorganization, Property 7: For any source file in the
    clean repo, the file contains no absolute machine-specific filesystem path
    (for example, no path beginning with a user home directory such as
    /home/<user>/).**

    Validates: Requirements 6.2
    """
    with open(path, encoding="utf-8") as handle:
        content = handle.read()

    for pattern in _HOME_PATH_PATTERNS:
        match = pattern.search(content)
        assert match is None, (
            f"{os.path.relpath(path, _CLEAN_REPO_ROOT)} contains an absolute "
            f"machine path: {match.group(0)!r}"
        )
