# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""create_server / stop_server: the non-blocking server factory the worker uses."""

from __future__ import annotations

import logging
import socket

import grpc
import pytest

from nxmndr.inference import inference_pb2, inference_pb2_grpc
from nxmndr.server import managers, server

pytestmark = pytest.mark.integration


def _accepts(host, port):
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        try:
            sock.connect((host, port))
        except OSError:
            return False
        return True


class _SpyServer:
    def __init__(self, inner, calls):
        self._inner, self._calls = inner, calls

    def stop(self, grace):
        self._calls.append(("grpc.stop", grace))
        return self._inner.stop(grace)


def test_create_server_binds_loopback_and_stop_server_stops_grpc_then_the_cache(
    tmp_path, monkeypatch, caplog
):
    srv, port, service = server.create_server(model_cache_dir=tmp_path / "cache", max_cores=1)
    calls = []
    try:
        assert isinstance(service, server.InferenceService) and port > 0
        assert _accepts("127.0.0.1", port)
        if socket.has_ipv6:
            assert not _accepts("::1", port)  # loopback IPv4 only, not every interface

        caplog.set_level(logging.DEBUG, logger="nxmndr.server")
        with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = inference_pb2_grpc.InferenceServiceStub(channel)
            secrets = {
                "authorization": "Bearer sekrit-bearer-1",
                "x-api-key": "sekrit-api-key-2",
                "hf-token": "sekrit-hf-token-3",
                "x-session-secret": "sekrit-4",
            }
            metadata = list(secrets.items()) + [("x-request-id", "visible-request-id")]
            assert stub.Health(inference_pb2.HealthRequest(), metadata=metadata).ready
            caps = stub.Capabilities(inference_pb2.CapabilitiesRequest())
        assert {c.key: c.value for c in caps.capabilities}["stream_context_version"] == "1"

        logged = [r.getMessage() for r in caplog.records if r.name == "nxmndr.server"]
        request_lines = [m for m in logged if "method=/nxmndr.inference" in m or "Health" in m]
        assert any("visible-request-id" in m for m in request_lines)  # the interceptor logged
        for value in secrets.values():
            assert all(value not in m for m in logged)
        assert all(name in "".join(request_lines) for name in secrets)  # keys stay visible

        real_shutdown = service.model_manager.shutdown

        def shutdown(*, drain_timeout_s=None):
            calls.append(("cache.shutdown", drain_timeout_s))
            real_shutdown(drain_timeout_s=drain_timeout_s)

        monkeypatch.setattr(service.model_manager, "shutdown", shutdown)
    finally:
        server.stop_server(_SpyServer(srv, calls), service, grace=1.5)

    assert calls == [("grpc.stop", 1.5), ("cache.shutdown", 1.5)]
    with pytest.raises(managers.CacheShutdownError):
        service.model_manager.acquire_execution(model_id="any")
    assert not _accepts("127.0.0.1", port)


def test_create_server_raises_when_the_port_is_taken(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        with pytest.raises(RuntimeError):
            server.create_server(port=taken.getsockname()[1], model_cache_dir=tmp_path / "c")
