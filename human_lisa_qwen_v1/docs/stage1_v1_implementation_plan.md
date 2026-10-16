# Stage1 v1 TDD Implementation Plan

## 1. Freeze current evidence

- Save a read-only audit manifest containing current git status, dataset
  counts, active GPU processes, old run arguments, and evaluator limitations.
- Do not overwrite old checkpoints or metrics.

## 2. Official-data adapter

Tests first:

- official counts and validity fields;
- crowd/invalid filtering, including annotation 2206849;
- deterministic person ordering and source-ann binding;
- box/keypoint/mask synchronized horizontal flip;
- missing gender/age field validity;
- non-crowd maxima 19 train and 13 val.

Implementation:

- add a COCO-UniHuman v1 dataset adapter that reads official JSON;
- decode polygons with pycocotools;
- return structured instances and validity masks;
- retain compact metadata for official result export.

## 3. Target serialization and collator

Tests first:

- exact special-token occurrence counts;
- multimodal prefix masking uses processor sequence positions;
- missing attribute value spans are CE-masked;
- dynamic padding works for batch size 1 and greater than 1;
- a 5001-token record raises instead of truncating;
- the maximum-person real sample is accepted at length 5000.

Implementation:

- replace silent truncation with tokenize-measure-reject;
- build task occurrence maps and target validity tensors;
- provide image-grid metadata and original/processed geometry;
- add deterministic worker seeding and pinned-memory controls.

## 4. Box and quality head

Tests first:

- normalized box conversion is valid and differentiable;
- L1/GIoU match known examples;
- token/target count mismatch raises;
- quality target and confidence composition are finite and ordered.

Implementation:

- replace xyxy-sorting sigmoid head with cxcywh regression;
- add IoU-quality head and generation-score composition;
- remove hard-coded score 1.0 from evaluation output.

## 5. Pose decoder

Tests first:

- output shapes for zero, one, and multiple persons;
- person/joint masks prevent cross-image leakage;
- invisible keypoints do not affect coordinate/OKS losses;
- a synthetic perfect prediction has zero L1 and near-zero OKS loss;
- token/target bindings survive missing pose fields.

Implementation:

- add record-conditioned 17-joint queries;
- add within-person, same-joint-across-person, and visual cross-attention;
- add coordinate, visibility, and OKS losses;
- export COCO keypoint detections and scores.

## 6. Segmentation decoder

Tests first:

- polygon/RLE decoding and transform alignment;
- Qwen grid tokens reshape to expected spatial maps;
- mask decoder supports multiple persons and variable grids;
- perfect masks minimize BCE/Dice;
- empty/missing masks are masked, not negative.

Implementation:

- expose selected Qwen visual features;
- add lightweight multiscale pixel decoder;
- add <seg> mask embedding and point-sampled BCE/Dice;
- export original-resolution COCO RLE masks.

## 7. Training engine

Tests first:

- parameter groups contain the intended LoRA/visual/head parameters;
- gradient accumulation performs exactly one optimizer step per four batches;
- warmup/cosine scheduler uses optimizer-step semantics;
- non-finite losses, gradients, and alignment failures stop the run;
- full checkpoint resume reproduces the next step.

Implementation:

- add scheduler, deterministic seeds, checkpoint state, task-valid logging,
  memory/token logging, validation hooks, and run manifest;
- keep physical batch 1, accumulation 4, BF16, max length 5000.

## 8. Guided inference and health monitor

