# Changelog

## Unreleased

### Breaking changes

Server and proxy:
- The gRPC server binds 127.0.0.1 by default (was every interface, `[::]`), and `nxmndr-server --host` now sets the gRPC address as well as the HTTP proxy's. Operator: pass `--host 0.0.0.0` to accept other machines, after reading [Security](README.md#security).
- The HTTP proxy (`nxmndr-server --azure-proxy`, `python -m nxmndr server`, `python -m nxmndr.server.azure_openai_proxy`) binds 127.0.0.1 by default (was 0.0.0.0). `python -m nxmndr grpc` gains `--host`. Operator: pass `--host 0.0.0.0` to accept other machines.
- For OpenAI-platform endpoints the proxy sends the configured deployment as `model` and replaces a `model` field from the client (was: the client's value was forwarded). Client: select the model with the deployment in the route.
