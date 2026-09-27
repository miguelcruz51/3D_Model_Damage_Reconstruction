"""Property-based test for docstring coverage on the public API (Property 9).

Introspects every public module, class, and function defined in the clean-repo
``src.models``, ``src.losses``, and ``src.data`` packages and asserts that each
has a non-empty docstring. "Public" means the name does not start with an
underscore, and the object is *defined in* one of these packages (imported
third-party symbols are ignored).
"""

import importlib
import inspect
import os
import pkgutil

from hypothesis import given, settings
from hypothesis import strategies as st

_TARGET_PACKAGES = ("src.models", "src.losses", "src.data")


def _iter_modules(package_name):
    """Yield imported module objects for a package and its submodules."""
    package = importlib.import_module(package_name)
    yield package
    if hasattr(package, "__path__"):
        for info in pkgutil.iter_modules(package.__path__):
            if info.name.startswith("_"):
                continue
            yield importlib.import_module(f"{package_name}.{info.name}")


def _collect_public_api():
    """Return (label, object) pairs for public modules/classes/functions.

    Only classes and functions *defined in* a target package are included, so
    re-exported third-party objects are not held to the docstring requirement.
    """
    items = []
    target_prefixes = tuple(_TARGET_PACKAGES)
    for package_name in _TARGET_PACKAGES:
        for module in _iter_modules(package_name):
            items.append((f"module {module.__name__}", module))
            for name, obj in vars(module).items():
                if name.startswith("_"):
                    continue
                if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                    continue
                obj_module = getattr(obj, "__module__", "") or ""
                if not obj_module.startswith(target_prefixes):
                    continue
                kind = "class" if inspect.isclass(obj) else "function"
                items.append((f"{kind} {module.__name__}.{name}", obj))
    # De-duplicate by label (same object re-exported across modules).
    unique = {}
    for label, obj in items:
        unique.setdefault(label, obj)
    return sorted(unique.items(), key=lambda pair: pair[0])


# Ensure clean_repo root is importable (conftest also does this).
_CLEAN_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys  # noqa: E402

if _CLEAN_REPO_ROOT not in sys.path:
    sys.path.insert(0, _CLEAN_REPO_ROOT)

_PUBLIC_API = _collect_public_api()


@settings(max_examples=100, deadline=None)
@given(entry=st.sampled_from(_PUBLIC_API))
def test_docstring_coverage(entry):
    """**Feature: repo-reorganization, Property 9: For any public module, class,
    or function defined in the clean repo src/models, src/losses, or src/data
    packages, a docstring is present.**

    Validates: Requirements 12.1
    """
    label, obj = entry
    doc = inspect.getdoc(obj)
    assert doc is not None and doc.strip(), f"missing docstring: {label}"
