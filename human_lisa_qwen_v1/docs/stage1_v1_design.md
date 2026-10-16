# COCO-UniHuman v1 Stage1 Stable Baseline Design

## Scope

Stage1 trains Qwen2.5-VL-3B to emit a variable-length, person-bound JSON
record list. Each record owns one <box> token and, when the corresponding
annotation is valid, one <pose> and one <seg> token. Token hidden states
drive discriminative task decoders while the language model supervises record
count, ordering, structure, gender, and age.

This is the stable baseline. A fixed K=25 person-query variant is intentionally
deferred to a later experiment.

## Authoritative data

- Train annotations:
  /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json
- Validation annotations:
  /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json
- Images:
  /share2/program/weiminghong/coco/train2017 and val2017

Verified counts:

| Split | Images | Instances | Non-crowd max/image | Box | Pose | Seg valid | Gender valid | Age valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 64,115 | 262,465 | 19 | 262,465 | 262,465 | 262,465 | 52,812 | 53,542 |
| val | 2,693 | 11,004 | 13 | 11,004 | 11,004 | 0 | 1,970 | 1,998 |

Validation polygons are present for every annotation despite
valid.segmentation=0; the HumanQueryNet/COCO evaluator uses these polygons for
segmentation evaluation. Training validity follows the valid dictionary.

Dataset rules:

1. Exclude iscrowd=1, ignored, non-positive-area, and malformed boxes.
2. Preserve the original annotation id as source_ann_id.
3. Sort records by box center left-to-right, then top-to-bottom.
4. Reassign contiguous output person_id values only after filtering/sorting.
5. Apply one geometric transform to image, box, keypoints, and mask together.
6. Re-sort records after horizontal flip/crop.
7. Keep per-field validity; never turn a missing field into a negative label.
8. The one zero-height training annotation (id=2206849) is filtered.

## Output contract and binding

The assistant emits JSON only, with one record such as:

    {"person_id":1,"box":"<box>","pose":"<pose>","seg":"<seg>","gender":0,"age":31}

- The number of records is variable and generation ends with JSON closure/EOS.
- No prompt or schema imposes a maximum person count.
- The Nth <box>, <pose>, and <seg> occurrences belong to the Nth record,
  but task occurrence indexes are tracked explicitly because a field may be
  null.
- Teacher-forced training must assert exact equality between every emitted
  task token count and its valid target count. Silent min alignment is
  forbidden.
- Missing gender/age values remain -1 in the serialized target, but only the
  value token span is masked from CE. JSON punctuation and field names remain
  supervised.
- The guided JSON schema validates syntax at inference. Semantic validation
  separately checks monotonic unique person ids, record count, token/head
  count, finite outputs, box geometry, and EOS termination.

## Length policy

- max_length=5000.
- Processor tokenization always runs once with truncation=False.
- If any sequence is longer than 5000, training raises an error containing the
  image id, person count, prompt tokens, target tokens, visual tokens, and total
  tokens.
- Padding to 5000 is forbidden; dynamic per-batch padding is used.
- The current hook protocol's verified worst training example is about 1,418
  multimodal tokens before adding masks, so 5000 has ample margin.

## Model

### Qwen backbone

- Base: /share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct.
- BF16 forward/backward.
- LoRA targets both language attention and visual attention projections.
- Low-rank adapters on the token embedding and LM head learn <box>, <pose>,
  and <seg>; the full vocabulary matrices are not unfrozen.
- Gradient checkpointing is enabled; KV cache is disabled for training.

### Person record states

The final language hidden state at each special token is projected to 256
dimensions. The record-local <box> state is also the parent person state used
to condition the pose and mask decoders.

### Box and confidence

- A three-layer MLP predicts normalized cxcywh, converted to valid xyxy.
- Loss: L1 weight 5 and GIoU weight 2.
- A quality head predicts detached target IoU with BCE or quality-focal loss.
- Inference confidence is the geometric mean of:
  1. generated <box>/record anchor token probability, and
  2. predicted localization quality.
- COCO AP uses this confidence; hard-coded score 1.0 is forbidden.

### Multi-person pose decoder

For every valid `<pose>` occurrence, the decoder uses the full final spatial
Qwen visual-token map. It never crops an ROI and never perturbs a box.

1. Combine the record-local pose state, its parent `<box>` state, 17 learned
   joint embeddings, and a detached predicted-person box coordinate embedding.
2. Initialize one global-image joint reference per joint from learned canonical
   body coordinates transformed by that predicted person reference.
3. Perform two joint-query refinement layers. Each layer samples eight points
   from the full visual feature map around every evolving joint reference and
   predicts a residual coordinate update.
4. Predict normalized image coordinates and three-way keypoint visibility.

The same detached predicted person reference is used during training and
inference. Ground-truth boxes are supervision targets only; they are never
pose-decoder inputs. Thus pose has no ROI crop, bbox jitter, or GT-reference
teacher forcing.

Losses use valid keypoints only:

- normalized-image coordinate L1, weight 5;
- bounded 1-OKS loss, weight 1.5;
- visibility CE, weight 1.

The standard validation metric is COCO keypoint OKS AP/AP50.

### Segmentation decoder

- The final fused Qwen visual-token map is reshaped with image_grid_thw and
  refined by a lightweight convolutional pixel decoder.
- The <seg> state becomes a 256-D mask embedding.
- Mask logits are the mask-embedding/pixel-feature dot product.
- Dense resized-target BCE weight 8 and Dice weight 5 follow the HumanQueryNet
  baseline configuration.
- Masks are resized to original image coordinates for COCO RLE output.

### Record-level duplicate suppression

Guided JSON decoding has no fixed person-count cap. It may therefore emit
repeated but structurally valid person records. At inference, score-aware NMS at
IoU 0.90 suppresses only near-identical boxes. It operates on an assembled
person record, so the box, pose, mask and attributes of a retained person stay
bound together. Raw record count, suppressed IDs and raw semantic-health
failures are retained as diagnostics; the filtered records are sent to the
standard COCO and attribute evaluators.

### Attributes

Gender and age remain language-decoded record fields. CE is applied only where
the field is valid. The evaluator uses the official AttrCOCOeval behavior and
also reports coverage as a diagnostic.

## Optimization

- Physical batch size 1 and gradient accumulation 4.
- AdamW logs visual-LoRA, language/token-LoRA, box, pose, and mask learning
  rates; bias and normalization parameters have zero weight decay.
- Task-loss gradients entering the Qwen backbone are scaled by 0.1; head
  gradients remain unscaled.
- Box, pose, and mask heads use independent learning rates and norm clipping.
  The 8-image gate uses 5e-4, 1e-3, and 1e-4 respectively.
- The diagnostic gate uses 2% warmup, constant post-warmup LR, and per-family
- Full-run learning rates, scheduler, and clipping are selected only after the
  32/256-image smoke gates; the launch manifest records the selected values.
- One epoch is the first full run. Continue to later epochs only from official
  validation evidence, not training loss alone.
- Checkpoints save LoRA, task decoders, processor, optimizer, scheduler,
  RNG states, epoch, sampler position, and manifest. BF16 does not require a
  gradient scaler.

The legacy v2 diagnostic learned JSON and all three special tokens but produced
zero official pose AP even with GT records. It is retained as historical
diagnostic evidence, not as a candidate full-run configuration.

Every log record includes raw and weighted task losses, valid counts, learning
rates, gradient norm, token lengths, and memory. Non-finite loss/gradient or
zero-valid-task anomalies stop training.

## Evaluation

Primary full-val metrics:

- bbox: official COCO AP, AP50, AP75, APs/APm/APl;
- pose: official COCO keypoint OKS AP and AP50;
- segmentation: official COCO segm AP and AP50;
- gender/age: official HumanQueryNet AttrCOCOeval.

Diagnostics include schema validity, EOS rate, truncation count, GT/predicted
person counts, duplicate records/boxes, token/head occurrence mismatches,
per-size behavior, attribute coverage, latency, and tokens/image.

## Quality gates

1. Unit tests cover filtering, binding, transforms, exact prompt masking,
   overflow rejection, decoder shapes/losses, checkpoint state, and official
   evaluator golden cases.
2. An 8-image and then 32-image overfit run must decrease every valid task loss
   and produce visually aligned box/pose/mask outputs.
3. A 256-image smoke run must have zero truncations, non-finite values, silent
   target drops, and schema failures under guided decoding.
4. A fixed probe set is decoded periodically during training. Raw JSON,
   confidence, overlays, and health metrics are retained.
5. Full training starts only after gates 1-3 pass.

## Operational isolation

All code, logs, manifests, and checkpoints stay under
/home/weiminghong/humanslot and
/share2/program/weiminghong/humanslot_outputs. Existing Stage2 and unrelated
GPU processes are not modified.
