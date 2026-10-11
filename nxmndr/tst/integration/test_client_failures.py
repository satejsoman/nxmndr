# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""InferenceGrpcClient failure handling over a real loopback gRPC transport.

A failed call is retried only when the server could not be reached, and a failure
never ends the other calls of the same client.
"""

from __future__ import annotations

import threading

import grpc
import numpy as np
import pytest

from nxmndr.client import InferenceGrpcError
from tst.support.grpc_harness import running_server
from tst.support.stream_doubles import CountingLoader, GatedModel

pytestmark = pytest.mark.integration


class FailingModel:
    """Raises inside predict; the server answers INTERNAL."""

    def __init__(self):
        self.calls = 0

    def predict(self, arr, return_embeddings=False):
        self.calls += 1
        raise RuntimeError("inference failed")


@pytest.fixture
def failing():
    return FailingModel()


@pytest.fixture
def gated():
    return GatedModel()


@pytest.fixture
def broken_loads():
    return []


@pytest.fixture
def loader(failing, gated, broken_loads):
    def broken():
        broken_loads.append(1)
        raise RuntimeError("load failed")

    return CountingLoader(
        {
            "double://fail": lambda: (failing, "onnx"),
            "double://gated": lambda: (gated, "onnx"),
            "double://broken": broken,
        }
    )


def _load(client, source):
    return client.load_model("ignored", {"format": "onnx", "source": source, "task": ""})


def _chip():
    return np.ones((4, 4, 3), np.float32)


def test_a_failing_predict_runs_the_model_once(tmp_path, loader, failing):
    with running_server(tmp_path, loader=loader) as h, h.client(max_attempts=3, backoff_seconds=0) as c:
        mid = _load(c, "double://fail")
        with pytest.raises(InferenceGrpcError) as err:
            c.predict(mid, _chip())
        assert err.value.code == grpc.StatusCode.INTERNAL
        assert failing.calls == 1


def test_a_failed_call_does_not_end_a_stream_on_the_same_client(tmp_path, loader, gated, broken_loads):
    # The failing call is a LoadModel, so it needs no inference slot while the
    # stream's tile holds one.
    with running_server(tmp_path, loader=loader) as h, h.client(max_attempts=3, backoff_seconds=0) as c:
        gated_id = _load(c, "double://gated")
        call = c.stream_predict(model_id=gated_id, samples=[_chip()], tile_ids=["t0"])
        got, errors = [], []

        def consume():
            try:
                got.extend(call)
            except Exception as exc:  # surfaced by the assertion below
                errors.append(exc)

        consumer = threading.Thread(target=consume)
        consumer.start()
        assert gated.entered.wait(10), "the stream's tile never started"
        with pytest.raises(InferenceGrpcError) as err:
            _load(c, "double://broken")
        assert err.value.code == grpc.StatusCode.INTERNAL
        assert len(broken_loads) == 1
        gated.release.set()
        consumer.join(10)
        assert errors == [] and [r.metadata["tile_id"] for r in got] == ["t0"]


def test_a_failed_stream_does_not_end_a_predict_on_the_same_client(tmp_path, loader, gated):
    with running_server(tmp_path, loader=loader) as h, h.client(max_attempts=3, backoff_seconds=0) as c:
        gated_id = _load(c, "double://gated")
        result, errors = [], []

        def predict():
            try:
                result.append(c.predict(gated_id, _chip()))
            except Exception as exc:  # surfaced by the assertion below
                errors.append(exc)

        caller = threading.Thread(target=predict)
        caller.start()
        assert gated.entered.wait(10), "the predict never started"
        with pytest.raises(InferenceGrpcError) as err:
            list(c.stream_predict(session_id="never-opened", samples=[_chip()], tile_ids=["t0"]))
        assert err.value.code == grpc.StatusCode.NOT_FOUND
        gated.release.set()
        caller.join(10)
        assert errors == [] and tuple(result[0].shape) == (4, 4, 3)
