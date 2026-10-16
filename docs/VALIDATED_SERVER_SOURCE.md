# Retained server implementations

The current source tree selects implementations with recorded task-metric
improvement or sustained finite, decreasing training loss, plus their required
shared modules. This is an engineering retention rule, not a declaration that
all five tasks or the paper's scientific claims have succeeded.

## Included implementations

- `human_lisa_qwen_v3_repr`: completed two-epoch historical training, with all
  five generated-task AP values improving from epoch 1 to epoch 2 under the same
  recorded annotation, split, image count, scorer protocol, and score strategy.
- `human_lisa_qwen_v4`: sustained finite training-loss records and decreasing
  total-loss window means. This historical implementation still has weak dense
  tasks; it is not the later LISA/pose-routing mainline.
- `qwen_dinov3_hqnet_pipeline` and `HumanQueryNet_mmdet3_port`: the historical
  Stage-2 pipeline has finite, decreasing loss over its saved training records.
  Its run did not complete the planned dataset and has no qualified matched
  transfer conclusion.
- Native Qwen box/gender/age evaluation and required data/model/scoring helpers.
- Necessary shared modules in `human_lisa_qwen_v1` and `human_lisa_qwen_v2`,
  and server environment scripts. Their inclusion is dependency support, not
  promotion of every historical experiment in those versions.

The retained code passed 22 VLM component tests in the existing `human4b`
environment and 9 Stage-2 checks in the existing `human_stage2` environment.
Tests ran without training or model downloads. Stage-2 emitted existing syntax
and dependency deprecation warnings. No full model evaluation was rerun.

## Deferred source

Untrained v5, obsolete standalone alternatives, and unverified historical debug
launchers are omitted from this integration. The later routing implementation
`human_lisa_pose_stage1`, referenced by historical records, was not present in
the inspected server source tree; this integration does not recreate it or claim
that it is included.

## History and asset policy

The full original server snapshot remains at commit
`abd9e1d71ef365c25b93dfc23426fc917a4fe31b` on
`codex/server-code-sync-20261016`. Filtering is a new descendant commit; previous
commits are preserved. Pre-existing main files are retained unchanged except for
selected source updates and this documentation.

No datasets, annotation payloads, model weights, checkpoints, experiment output
files, prediction files, training logs, or caches are added by this integration.
Prior third-party fixtures and assets already tracked in the old main history
remain unchanged. This operation does not purge or rewrite that history.

Detailed selection and numerical audit records remain on the server outside
Git, under `/root/autodl-tmp/git-sync-20261016/`.
