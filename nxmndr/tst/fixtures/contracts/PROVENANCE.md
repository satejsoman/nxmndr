# Contract fixtures copied from the plugin

The backend repository cannot import the plugin's `contracts` package, so contract
fixtures are copied here verbatim, with the plugin commit they came from.
`tst/unit/test_stream_context_fixture.py` pins each copy's SHA-256; when the plugin
changes a fixture, copy it again and update this table and the pinned hash together.

| File | Source | Plugin commit | SHA-256 |
|---|---|---|---|
| `stream_context_v1.json` | `nxmndr-qgis` `src/contracts/fixtures/stream_context_v1.json` | `a1f353dce737de75e8e0a85a74f5cb050b4becf5` | `212eeb8207afc688c012ba56733824b1842b556c26c698e2841e8fa8f1466e06` |
