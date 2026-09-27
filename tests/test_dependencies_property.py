"""Property-based test for dependency import coverage (Property 8).

Parses every clean-repo source file under ``src/`` and ``scripts/`` for its
top-level imports, filters out standard-library and first-party modules, and
asserts that each remaining third-party top-level package is declared in
``clean_repo/requirements.txt``.
"""

import ast
import os
import sys

from hypothesis import given, settings
from hypothesis import strategies as st

_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SOURCE_DIRS = ("src", "scripts")

# First-party package roots that live inside clean_repo and are therefore not
# third-party dependencies.
_FIRST_PARTY = {"src", "scripts", "tests"}

# Map a top-level import name to the distribution name expected in
# requirements.txt when the two differ.
_IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
}


def _iter_source_files():
    """Yield absolute paths of every ``.py`` file under the source dirs."""
    for source_dir in _SOURCE_DIRS:
        root = os.path.join(_CLEAN_REPO_ROOT, source_dir)
        for dirpath, _dirnames, filenames in os.walk(root):
            if "__pycache__" in dirpath:
                continue
            for name in filenames:
                if name.endswith(".py"):
                    yield os.path.join(dirpath, name)


def _top_level_imports(path):
    """Return the set of top-level module names imported by ``path``."""
    with open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)

    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports (level > 0); they are first-party.
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _third_party_imports():
    """Collect all third-party top-level import names across the source tree."""
    stdlib = set(sys.stdlib_module_names)
    third_party = set()
    for path in _iter_source_files():
        for module in _top_level_imports(path):
            if module in stdlib or module in _FIRST_PARTY:
                continue
            third_party.add(module)
    return third_party


def _requirements_packages():
    """Return the set of lower-cased package names listed in requirements.txt."""
    req_path = os.path.join(_CLEAN_REPO_ROOT, "requirements.txt")
    packages = set()
    with open(req_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            # Strip version specifiers and extras, keep the bare name.
            name = line
            for sep in ("==", ">=", "<=", "~=", ">", "<", "[", ";"):
                name = name.split(sep, 1)[0]
            name = name.strip().lower()
            if name:
                packages.add(name)
    return packages


_THIRD_PARTY_IMPORTS = sorted(_third_party_imports())


@settings(max_examples=100, deadline=None)
@given(module=st.sampled_from(_THIRD_PARTY_IMPORTS))
def test_dependency_import_coverage(module):
    """**Feature: repo-reorganization, Property 8: For any third-party
    top-level package imported by a clean-repo source file, that package is
    listed in the clean repo requirements.txt.**

    Validates: Requirements 10.1
    """
    listed = _requirements_packages()
    distribution = _IMPORT_TO_DISTRIBUTION.get(module, module).lower()
    assert distribution in listed, (
        f"third-party import {module!r} (distribution {distribution!r}) is "
        f"not listed in requirements.txt; listed packages: {sorted(listed)}"
    )
