# Stage1 V3 Flat-Pose Architecture and Protocol

## Scope

This is the COCO-UniHuman v1 Stage1 baseline for five jointly trained outputs:

- native autoregressive text: person records, normalized integer box, gender, age;
- dense heads: COCO-17 pose and instance segmentation;
- shared binding: K=25 image-conditioned person queries matched with Hungarian assignment.

The Qwen3.5-4B visual tower and language model both receive all-linear LoRA updates in the main baseline. This covers every visual attention/MLP block, all gated-linear-attention and self-attention layers, and every language MLP layer. Attention-only LoRA is retained only as an explicit ablation. A startup audit hard-fails if the injected layer counts differ from the audited Qwen3.5-4B topology. Three newly added special-token embedding rows are trainable. The visual neck, person-query decoder, direct pose decoder, and mask decoder are fully trainable.

## Source-grounded design

| Work | Pinned source | Borrowed component | Deliberate deviation |
|---|---:|---|---|
| COCO-UniHuman / HumanQueryNet | `7b5f57b` | person-query set prediction, Hungarian assignment, focal/L1/GIoU weights, task-specific missing-label masking, official AttrCOCOeval | K=25 follows the audited v1 maximum (train 19, val 13), not the original 300-query detector |
| GroupPose | `127fe0e` | COCO-17 target semantics, OKS supervision, deep auxiliary supervision pattern | reference/attention pose refinement failed the real 4-image memorization gate, so the baseline uses direct coordinates |
| LISA | `3cb2d43` | task-token hidden state as a dense-decoder condition | one mask is produced per matched person query, not one conversational mask |
| PixelLM | `ebf0792` | person-aware token-to-mask conditioning and multi-person dense output | shared person query supplies binding instead of free token order alone |
| PaDT | `912a8ae` | explicit human/task association as a first-class design constraint | association is implemented by set matching plus one atomic person record |
| Qwen-VL grounding | local Qwen3.5-4B | normalized autoregressive text coordinates | standard COCO AP confidence comes from matched query objectness; constant-score AP remains a diagnostic |

## Data and binding

1. Non-crowd valid persons are sorted by full-body box center from left to right, then top to bottom.
2. The person-query decoder predicts 25 objectness scores and normalized `cx,cy,w,h` boxes from Qwen visual feature maps.
3. Hungarian cost uses focal class cost 1, box L1 cost 5, and GIoU cost 2.
4. Matched queries are reordered to the GT person order for teacher forcing.
5. The same matched query conditions pose and mask outputs for that person.
6. Missing pose or segmentation labels suppress only that task loss. They do not create negatives and do not remove box or attribute supervision.
7. COCO-UniHuman v1 train marks segmentation valid and supplies masks. The v1 val JSON retains raw COCO person polygons although its segmentation-valid flag is zero; standard segm COCOeval, including HumanQueryNet's evaluate-all path, evaluates those raw polygons. This task is binary person instance segmentation, not part-level human parsing.

No GT box, jittered box, ROI crop, heatmap, or pose reference point is supplied to the pose decoder.

## Decoders and losses

- Text: causal CE with explicit task spans; unknown gender/age tokens are masked.
- Query detector: sigmoid focal objectness, normalized box L1, GIoU, and intermediate-layer auxiliary loss.
- Pose: matched visual query plus `<person>` and `<pose>` hidden states, followed by a two-layer MLP that directly predicts 17 normalized XY pairs and 17 three-way visibility logits. Losses are visible-joint L1, COCO OKS, visibility CE, and intermediate-coordinate auxiliary L1.
- Segmentation: matched visual query plus `<person>` and `<seg>` hidden states condition a LISA/PixelLM-style dynamic mask embedding. Losses are pixel BCE and Dice.

Default paper-style weights are detector `1/5/2`, pose `10/4/1/0.5`, and mask `2/5`. Total loss magnitude is not treated as an accuracy metric; every raw component and every trainable-family gradient norm is logged separately. Gradient accumulation is closed at every epoch boundary, so each epoch checkpoint is taken after an optimizer step and is independently recoverable even when the number of micro-batches is not divisible by the accumulation factor.

## Structured generation

Training and inference use the same JSON person-record schema. Inference uses `lm-format-enforcer` prefix-constrained JSON decoding, greedy generation, EOS checks, exact schema validation, and a hard `5000` new-token cap. Training uses `truncation=False`; any sequence over 5000 raises `SequenceOverflowError` with image, person, and token counts.

Schema validity does not imply semantic correctness. Evaluation separately reports raw schema validity, repaired person IDs, EOS rate, token-cap hits, duplicate suppression, person-count error, and official task metrics.

## Evaluation

- Box: official COCO `bbox` AP/AP50/AP75 on native text boxes, ranked by matched query objectness.
- Pose: official COCO keypoint OKS AP/AP50/AP75.
- Segmentation: official COCO mask AP/AP50/AP75.
- Gender and age: HumanQueryNet AttrCOCOeval with partial-label filtering.
- Diagnostic controls: constant-score official metrics, teacher-forced dense metrics, query-box match IoU, and schema/semantic-health counters.

All tasks remain one atomic person record at inference. Duplicate suppression is score-aware and only removes near-identical boxes at IoU >= 0.90; it is not a person-count cap.

## Mandatory gates before full training

1. Static compile/import and exact token/data audit.
2. One real-image forward/backward step with exact LoRA layer-coverage checks and finite, nonzero gradients for all seven parameter families.
3. Four-image joint overfit with query, pose, mask, and text improvements.
4. Strict generated JSON plus dense decode on the same four images.
5. A 256-image smoke run covering missing labels and crowded examples.
6. Batch-size and throughput benchmark on the A800.
7. Full 64,115-image, three-epoch training with per-task losses and epoch checkpoints; every epoch checkpoint receives the same full-val evaluator. Epoch checkpoints include adapter/task heads, optimizer, scheduler, and RNG states, and are recoverable only under an exact manifest match.
## BoxPose Patch Audit

This isolated `human_lisa_qwen_v3_boxpose` branch changes the Stage1 pose decoder from a global normalized coordinate MLP to a box-relative decoder.
The `<pose>` hidden states still produce 17 COCO keypoints, but coordinates are decoded inside an expanded matched person-query box: `center + (sigmoid(local)-0.5) * box_size * 1.5`, clamped to image-normalized coordinates.
This uses the model-predicted matched query box during train/inference and does not use GT-box jitter.

Formal full retraining must use fresh checkpoints from this branch; old flatpose checkpoints are not architecture-equivalent for pose.
