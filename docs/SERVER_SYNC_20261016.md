# Server source snapshot: 2026-10-16

Source: `/root/autodl-tmp/humanslot` on the existing AutoDL server.
Base: GitHub main `5c56cb2c3326e62e4ac4f0c631bd480d475f7ada`.
Branch: `codex/server-code-sync-20261016`.

This additive source sync preserves all pre-existing repository files and history.
Files present on the server replace their corresponding files only in this branch;
previous versions remain available in the base commit. Files absent on the server
are retained. The original server source directory is unchanged.

Includes project source, tests, configs, documentation, research plans, and server
setup/transfer shell scripts under `server_support/`. These support scripts retain
their original server-specific paths and require inspection before use elsewhere.

Excludes datasets, annotations, model weights, checkpoints, experiment outputs,
generated experiment state/audits, logs, caches, local IDE metadata, media assets,
and editor backups. Existing assets already tracked in the base repository remain
unchanged; this sync neither reuploads nor removes them. Runtime research-harness
state files remain on the server and are not bundled in this source snapshot.

Validation: content hashes checked against source files, staged file exclusions
checked, and credential patterns scanned. No training or evaluation was launched;
this snapshot does not assert that all historical experimental branches run.
