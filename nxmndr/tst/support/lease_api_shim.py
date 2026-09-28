# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Install the in-test lease API fake while chunk 1a's managers.py is not merged.

Integration point: server.py imports the frozen lease API names from
``nxmndr.server.managers``. If every name already exists (chunk 1a merged), this
does nothing and the tests run against the real manager.

``nxmndr/server/__init__.py`` imports ``server.py`` eagerly, and ``server.py``
imports the lease names from ``managers`` at import time, so the names must be
added while ``managers`` is first executed. ``install()`` therefore registers an
import hook that completes the module right after its own code runs.
"""

from __future__ import annotations

import importlib.abc
import sys

from . import lease_manager_fake as _fake

MANAGERS_MODULE = "nxmndr.server.managers"
REQUIRED_NAMES = tuple(_fake.__all__)


def _complete(module) -> bool:
    """Add the fake names to a managers module that lacks the lease API."""

    if getattr(module, "_LEASE_API_FAKE_INSTALLED", False):
        return True
    if all(hasattr(module, name) for name in REQUIRED_NAMES):
        return False
    _fake.LEGACY_MODEL_MANAGER = module.ModelManager
    for name in REQUIRED_NAMES:
        setattr(module, name, getattr(_fake, name))
    module._LEASE_API_FAKE_INSTALLED = True
    return True


class _CompletingLoader(importlib.abc.Loader):
    def __init__(self, real_loader):
        self._real = real_loader

    def create_module(self, spec):
        return self._real.create_module(spec)

    def exec_module(self, module):
        self._real.exec_module(module)
        _complete(module)


class _ManagersFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != MANAGERS_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                spec.loader = _CompletingLoader(spec.loader)
                return spec
        return None


def install() -> None:
    """Make ``nxmndr.server.managers`` expose the lease API (fake only if missing)."""

    module = sys.modules.get(MANAGERS_MODULE)
    if module is not None:
        _complete(module)
        return
    if not any(isinstance(f, _ManagersFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _ManagersFinder())


def fake_active() -> bool:
    """True when the tests run against the in-test fake rather than chunk 1a's manager."""

    import nxmndr.server.managers as managers

    return bool(getattr(managers, "_LEASE_API_FAKE_INSTALLED", False))
