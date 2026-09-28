# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tiny PyTorch models and RPC worker targets for the PyTorch RPC service tests.

They live in an importable module because the ``spawn`` worker unpickles model
classes and process targets by reference.

``AddConst`` returns its input plus a scalar ``bias`` read from the weights file, so
two weights files give two models with distinguishable outputs from one class.
The ``*_worker`` functions replace ``RpcWorkerManager.worker_target`` to make the
child fail before it is ready to join the RPC group.
"""

from __future__ import annotations

import os
import time

import torch

ADD_CONST_CLASS = "nxmndr_test_add_const"


class AddConst(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return x + self.bias


def save_add_const(path, bias: float) -> str:
    torch.save({"bias": torch.tensor(float(bias))}, str(path))
    return str(path)


def register() -> None:
    from nxmndr.models import register_pytorch_model

    register_pytorch_model(ADD_CONST_CLASS, AddConst)


def crashing_worker(master_addr, master_port, ready_port, startup_timeout_s):
    """Exits with code 3 before it reports ready (a broken worker environment)."""

    os._exit(3)


def silent_worker(master_addr, master_port, ready_port, startup_timeout_s):
    """Never reports ready (a hung import); exits only when its parent is gone."""

    parent = os.getppid()
    deadline = time.monotonic() + 600
    while os.getppid() == parent and time.monotonic() < deadline:
        time.sleep(0.1)
