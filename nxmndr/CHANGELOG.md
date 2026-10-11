# Changelog

## Unreleased

### Breaking changes

Server and proxy:
- The gRPC server binds 127.0.0.1 by default (was every interface, `[::]`), and `nxmndr-server --host` now sets the gRPC address as well as the HTTP proxy's. Operator: pass `--host 0.0.0.0` to accept other machines, after reading [Security](README.md#security).
- The HTTP proxy (`nxmndr-server --azure-proxy`, `python -m nxmndr server`, `python -m nxmndr.server.azure_openai_proxy`) binds 127.0.0.1 by default (was 0.0.0.0). `python -m nxmndr grpc` gains `--host`. Operator: pass `--host 0.0.0.0` to accept other machines.
- For OpenAI-platform endpoints the proxy sends the configured deployment as `model` and replaces a `model` field from the client (was: the client's value was forwarded). Client: select the model with the deployment in the route.

Client (`nxmndr.client`):
- `InferenceGrpcClient` retries a unary call only when it fails with UNAVAILABLE (was every status code). A call that fails with INTERNAL or another server answer, or whose deadline passes, raises `InferenceGrpcError` after one attempt. Client: give slow calls (an `open_session` or `load_model` that downloads a model) a longer `timeout`, and retry a failed call yourself only where running it again is safe.
- An error no longer closes the client's channel (was: every failed call or stream closed it, which also ended every other call in progress on the same client). After a server outage the channel reconnects with gRPC's backoff, which can take several seconds. Client: call `close()` to open a new channel on the next call at once.
