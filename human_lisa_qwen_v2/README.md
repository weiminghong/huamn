# Human LISA Qwen Stage1 V2

Stage1 V2 is an isolated replacement for the V1 baseline. It keeps native Qwen
text generation for person count, boxes, gender, and age, and uses token states
plus high-resolution Qwen vision features for pose and person segmentation.

## Person record

```json
{"persons":[{"person_id":1,"person":"<person>","pose":"<pose>","seg":"<seg>","bbox":[120,80,640,980],"gender":0,"age":31}]}
```

Every generated person owns one `<person>`, one `<pose>`, and one `<seg>` state.
Partial labels mask the corresponding head loss; they do not teach the language
model to omit a task token. Persons are sorted by body-box center and rebound to
contiguous IDs.

The pose and segmentation anchors precede the text box in causal order. Dense
heads therefore cannot consume the current record's box coordinates. There is
no RoIAlign, predicted-box reference, GT-box reference, or box jitter path.

## Related implementations used

| Work | Commit | Principle retained | Deliberate boundary |
| --- | --- | --- | --- |
| GroupPose | `127fe0e` | COCO-17 joint queries, within-person reasoning, same-joint cross-person interaction, L1 + OKS + visibility supervision | Stage1 uses variable generated person records, not fixed K queries or Hungarian matching |
| LISA | `3cb2d43` | task token hidden state drives a mask decoder; BCE + Dice supervision | Qwen raw vision features replace a separately frozen SAM encoder |
| PixelLM | `ebf0792` | multiple vision depths and a higher-resolution pixel decoder | one mask token per person is used for the baseline |
| HumanQueryNet | `7b5f57b` | one atomic person identity binds all tasks; sparse labels are loss-masked | no old MMDetection runtime is imported into Stage1 |
| PaDT | `912a8ae` | language-side person anchors remain grounded in image features | no paper-specific visual-reference tokenizer is added |

Reference repositories are pinned under
`/root/autodl-tmp/humanslot_refs/stage1_v2_20260723`.

## Architecture

1. Qwen3.5 produces the language hidden states and raw unmerged vision states.
2. Raw vision layers 8, 16, and 24 are restored from block-major order to the
   H-by-W patch grid.
3. A trainable neck fuses the three depths and builds three spatial levels.
4. `<person> + <pose>` expands to 17 joint queries. Two decoder layers apply
   within-person attention, same-joint attention across people, and full-image
   cross-attention before global coordinate and visibility prediction.
5. `<person> + <seg>` cross-attends to the image and produces a dynamic mask
   embedding over a stride-4 pixel map.
6. Text bbox coordinates are normalized integers in 0..1000 and are supervised
   only by causal language CE.

## Trainable parameters

- LoRA rank 16 on every Qwen vision block `attn.qkv/attn.proj`.
- LoRA rank 16 on matching Qwen language attention projections.
- Only the three reserved vocabulary rows for `<person>`, `<pose>`, and `<seg>`.
- The visual neck, pose decoder, and mask decoder.

The three reserved rows are initialized from existing `person`, `pose`, and
`segmentation` embeddings. The original 248320-row model vocabulary is never
shrunk. Frozen base parameters use BF16 and trainable parameters remain FP32.

## Loss and optimization

Default full-run weights:

```text
CE 1.0
pose visible L1 10.0
pose OKS 4.0
pose visibility CE 1.0
pose auxiliary L1 0.5
mask BCE 2.0
mask Dice 0.5 (must pass the mask gate before the full run)
```

Training uses one forward and one backward pass, global gradient clipping,
AdamW, and one continuous warmup-cosine schedule across all three epochs.
`max_length` is exactly 5000, processor truncation is disabled, and any overflow
raises an exception with image/person/token counts.

The model requests Qwen logits only at supervised assistant positions through
`logits_to_keep`; dense hidden states and all losses are unchanged.

## Validation gates

1. Compile and tensor tests.
2. Real-image forward/backward: all six parameter families must have finite,
   nonzero gradients.
3. 32-image overfit: teacher-forced pose PCK and mask mIoU/Dice must improve.
4. 256-image joint smoke with the exact full-run optimizer and augmentation.
5. Save/load round trip plus guided generation on validation images.
6. Full 64,115-image, three-epoch run.

A failed gate blocks the full run.

## Evaluation

`eval_stage1_v2.py` performs end-to-end guided generation and reports official
COCO bbox/keypoint/segmentation AP plus partial-label HumanQueryNet attribute
metrics. Native text boxes use constant score 1.0; token probability is stored
only as a diagnostic.

`eval_teacher_forced_v2.py` is a diagnostic head oracle. It holds GT person
records fixed and reports pose PCK and segmentation mIoU/Dice. These numbers are
not end-to-end leaderboard results.
