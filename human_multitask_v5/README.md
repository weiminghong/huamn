# Human Multi-task v5

This is an isolated design and implementation package. It does not modify or
resume the running v4/Stage-1 experiment.

## Scientific role

Stage-1 jointly adapts Qwen Vision and the Qwen language model to human images.
Box, named pose coordinates, age, gender, attributes and relations use normal
language decoding. Person segment and instance-level parsing use shared
Mask2Former modules conditioned by person-owned LLM routing states. Qwen Vision
is the only RGB encoder in the mainline.

Stage-2 compares DINOv3 alone, DINOv3 plus frozen base-Qwen visual features and
DINOv3 plus frozen v5 visual features. Stage-3 adds frozen image-conditioned
person/task states from v5. Data, DINO initialization, task decoders and training
budget stay matched across the transfer arms.

## Dataset contract

Every JSONL record is image-centered and contains deterministic person slots,
scene entities and person-grounded relation edges. Every label has an explicit
valid flag and provenance. Missing labels are never converted to negatives.

COCO-17 and MHP-16 poses map into a named 21-joint union with a per-joint schema
mask. COCO binary person masks and MHP-59 parsing maps remain distinct tasks.
PSG/VAW annotations may attach to a COCO person only after an audited
mutual-best box/mask match; ambiguous joins are rejected.

## Sampling

Training chooses a task first and then a dataset with probability proportional
to the square root of eligible record count. A packet is eligible only when its
requested supervision is present. Joint packets are retained as a controlled
fraction so the model also learns shared Human Slot ownership.

## Evaluation

COCO-UniHuman uses the existing no-small paper-aligned evaluator. MHP parsing,
MHP pose, PSG relations and VAW attributes use their own official protocols.
Metrics from different scorers or units are never averaged into a single AP.

## Current readiness

COCO-UniHuman assets are available. MHP v2, PSG and VAW are not present on this
server, and only 7.8 GB is currently free on the data disk. The package can be
unit-tested now, but a multi-source training launch is blocked until server-side
storage and official dataset assets are resolved and audited.
