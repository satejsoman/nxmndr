# Changelog

## Unreleased

### Breaking changes

Server and proxy:
- The gRPC server binds 127.0.0.1 by default (was every interface, `[::]`), and `nxmndr-server --host` now sets the gRPC address as well as the HTTP proxy's. Operator: pass `--host 0.0.0.0` to accept other machines, after reading [Security](README.md#security).
- The HTTP proxy (`nxmndr-server --azure-proxy`, `python -m nxmndr server`, `python -m nxmndr.server.azure_openai_proxy`) binds 127.0.0.1 by default (was 0.0.0.0). `python -m nxmndr grpc` gains `--host`. Operator: pass `--host 0.0.0.0` to accept other machines.
- For OpenAI-platform endpoints the proxy sends the configured deployment as `model` and replaces a `model` field from the client (was: the client's value was forwarded). Client: select the model with the deployment in the route.

Model loading (`LoadModel`, `UnloadModel`):
- `LoadModel` with a spec equal to that of a loaded model returns that model's `model_id` (`effective_metadata` has `model_cache_hit=true`) and loads no second copy (was: every `LoadModel` loaded a new copy under a new `model_id`). `UnloadModel` releases one `LoadModel` reference; the model is freed when every caller that loaded it has unloaded it and no session or call uses it (was: `UnloadModel` freed the model at once). Client: call `UnloadModel` once for each successful `LoadModel`.
- The server keeps at most `NXMNDR_MODEL_CACHE_CAPACITY` models (default 10). A new model at full capacity evicts the least recently used model that no session or call uses, also one whose `LoadModel` callers have not unloaded it, and `Predict` with its `model_id` then fails with `NOT_FOUND`. If every loaded model is in use, `LoadModel` fails with `RESOURCE_EXHAUSTED` (was: models stayed loaded until `UnloadModel`). Client: on `NOT_FOUND` call `LoadModel` again, or run tiles through a session (`OpenSession`), which keeps its model loaded until the session closes.
- `LoadModel` with `overwrite=true` fails with `FAILED_PRECONDITION` while a session, a running call or an earlier `LoadModel` caller holds the loaded model (was: `overwrite` was ignored and every `LoadModel` loaded a new copy). Client: unload your own `model_id` first; the next `LoadModel` then loads a new copy unless another caller still holds the model.

Sessions and streaming (`OpenSession`, `StreamPredict`):
- A session idle longer than `NXMNDR_SESSION_TTL_SECONDS` (default 3600) is expired by a sweep that runs every 60 seconds from server start, and its model is unpinned (was: idle sessions expired only when another `OpenSession` arrived, at most once a minute). A stream on an expired session ends with a stream-scope error (`error_code` `session_not_open`), and `CloseSession` reports status `expired`. A session with a running stream is never idle. Client: open a new session after a long pause, or raise the TTL.

Client (`nxmndr.client`):
- `InferenceGrpcClient` retries a unary call only when it fails with UNAVAILABLE (was every status code). A call that fails with INTERNAL or another server answer, or whose deadline passes, raises `InferenceGrpcError` after one attempt. Client: give slow calls (an `open_session` or `load_model` that downloads a model) a longer `timeout`, and retry a failed call yourself only where running it again is safe.
- An error no longer closes the client's channel (was: every failed call or stream closed it, which also ended every other call in progress on the same client). After a server outage the channel reconnects with gRPC's backoff, which can take several seconds. Client: call `close()` to open a new channel on the next call at once.
