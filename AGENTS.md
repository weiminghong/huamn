# Human Slot Project Instructions

Read research_harness/paper_goal.json, requirements.json, state.json,
stage0_stage1_readiness.json, evidence_ledger.json, and the active branch before
experiments. Run scripts/validate_human_slot_harness.py before launch.

## Active user-approved Stage-1

Qwen text decoding emits per-person JSON slots with box, age, gender, <pose>,
and <seg>. The routing-token hidden states condition Qwen-native segmentation
and pose decoders. Qwen Vision is the only mainline image encoder. External SAM
or ViTPose image encoders are controls only. Stage-1 must jointly adapt Vision,
the merger, and LLM; the latest explicit parameter contract is Vision and LLM
LoRA plus fully trained merger, routing rows, projectors, and dense modules.

The current mask implementation is a lightweight SAM-style decoder, not the
official SAM decoder or its pretrained weights. Pose uses ROI features and
SimCC. Do not claim these implementations reproduce LISA or ViTPose exactly.

## Current execution requirement

The user requested the full training run, not another 64-image pilot. Use every
COCO-UniHuman training image with all valid task labels and all person sizes.
Retain missing-label masks. Record dataset size, epochs, optimizer updates,
learning rates, and actual parameter updates in the run manifest and logs.

Small-sample accuracy gates remain diagnostics and must not silently prevent
this user-authorized full run. Numerical, label, ownership, gradient-path, and
resource-safety failures still require correction. Do not mark a failed gate
passed, and do not treat training/GT-box metrics as free-generation AP.

## Three-stage research object

Stage-2 freezes Stage-1 Vision and transfers its features to DINOv3, which also
receives RGB. Stage-3 freezes the entire Stage-1 VLM and additionally transfers
image-conditioned decoder states to the DINO-side model. The contribution is
progressive visual and semantic transfer, not a new mask or pose architecture.

Stage-1 qualification still requires generated person slots, generated-box
decoding, official no-small COCO-UniHuman evaluation, and visual transfer tests.
Starting the requested training is not evidence that these criteria are met.

## Server operation

Work on this server only; do not stage scripts or assets locally. Use the shared
.gpu0_exclusive.lock for GPU launches. Keep resumable periodic checkpoints and
exact launch scripts. Do not interfere with other experiments or alter datasets
and base checkpoints. Record training phase explicitly in every status report.

## Evaluation cadence

The active training run is evaluated once per fully completed epoch, not every
checkpoint-save step. A user-stopped one-off evaluation retains its partial
outputs and must not automatically resume. The epoch_evaluation_queue controller
pins exact boundary checkpoints
and serializes all evaluators with .gpu0_evaluation.lock. Saving every 500 steps
is for recovery only and does not trigger evaluation.


## Successor v5 design

human_multitask_v5 is an isolated successor candidate for COCO-UniHuman,
MHP-v2, PSG, and VAW supervision. It adds parsing, richer attributes, and
person-grounded relations while preserving explicit missing-label masks.

The currently running E-branch remains untouched and is a historical control.
Do not launch v5 until official source assets, server storage, annotation
decoders, split de-duplication, and scorer parity pass. The v5 mainline may use
the official Transformers Mask2Former pixel and masked-attention decoder modules
only with Qwen Vision features as their sole RGB-derived input. A complete
Mask2Former/Swin or SAM image encoder remains an upper-bound control.

COCO-17 and MHP-16 joints must be mapped by explicit names into the canonical
21-joint union. COCO binary person masks and MHP-59 parsing labels remain
different tasks. Unlabeled tasks are absent supervision, never negatives.
