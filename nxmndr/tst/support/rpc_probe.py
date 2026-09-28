# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""A model class that proves a spawned PyTorch RPC worker ran its target.

The RPC worker constructs ``model_spec.model_class()`` before anything else. This
class writes the file named by ``NXMNDR_TEST_RPC_MARKER`` and ends the child
process there, so no torch RPC rendezvous is attempted. It lives in an importable
module because a ``spawn`` child unpickles the spec by reference.
"""

from __future__ import annotations

import os
from pathlib import Path

MARKER_ENV = "NXMNDR_TEST_RPC_MARKER"


class MarkerModel:
    def __init__(self):
        Path(os.environ[MARKER_ENV]).write_text(f"pid={os.getpid()}", encoding="utf-8")
        raise SystemExit(0)
