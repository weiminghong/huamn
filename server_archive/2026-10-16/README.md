# Server records archive: 2026-10-16

This branch preserves research records before the AutoDL instance is deleted.
The main branch contains the selected implementations; the complete latest source
snapshot remains at `abd9e1d71ef365c25b93dfc23426fc917a4fe31b` on
`codex/server-code-sync-20261016`. This archive adds distinct historical source
variants, launchers, source backups, metrics, training logs, experiment contracts,
failed-run diagnostics, environment package versions, source reference commits,
and the results review. Historical failures are retained as evidence, not as
working mainline implementations.

Original server paths are preserved beneath this directory. MANIFEST.json records
source paths, byte sizes and SHA-256 hashes. Files are copied byte-for-byte.

Explicit exclusions: datasets and annotation subsets, model weights and experiment
checkpoints, tokenizers/processor payloads, binary outputs, and per-sample
predictions (including decode probes). Paths and hashes referring to excluded
assets may appear in provenance records; their contents are not included.
The user explicitly chose not to preserve experiment weights. Deleting the
instance also discards the excluded predictions, so future rescoring cannot rely
on this archive alone. The checkpoint path for historical v3_repr epoch 2 was
already absent when audited; this archive does not recover that missing model.

No secrets or private keys are intentionally included. No server instance or
original scientific output was deleted by this archival operation.
