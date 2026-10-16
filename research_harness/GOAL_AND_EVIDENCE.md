# Human Slot Goal and Evidence

Updated: 2026-08-30

## Goal

Test whether LISA-style person-specific pose and segment routing can teach Qwen human visual structure and person semantics that transfer into DINOv3.

## Three-Stage Story

1. Stage-1 uses the native Qwen text decoder for Human Slots, boxes, age, gender, <POSE>, and <SEG>. Routing states condition mature pose/mask decoders over Qwen Vision features.
2. Stage-2 transfers frozen Stage-1 Qwen visual features into DINOv3.
3. Stage-3 adds frozen routing-token and person-slot LLM states.

## Official Stage-1 Outputs

- Box, Gender, Age: parsed from free Qwen text generation.
- Segment: SAM-style decoder conditioned by the generated <SEG> state and Qwen Vision features.
- Pose: SimCC decoder conditioned by the generated <POSE> state, generated box, and Qwen Vision ROI features.
- GT boxes and specialist image encoders are diagnostics only.

## Required Evidence

- Five-task no-small AP and text-format validity.
- Generated-box owner binding.
- Dense gradient flow into Qwen Vision LoRA.
- Base versus adapted frozen mask/keypoint probes.
- Qwen Vision frozen and detached-feature ablations.
- Exact LISA and ViTPose/RTMPose upper-bound controls.
- Image and person-slot shuffle sensitivity.

## Evidence Boundary

The design follows LISA and mature pose decoding, but it is not implemented. It has no Stage-1 accuracy or visual-learning evidence yet. C1-C3 remain pending.

## Current Decision

Active mainline: E_STAGE1_QWEN_NATIVE_LISA_POSE_ROUTING_V1

Next action: A005_FREEZE_LISA_POSE_ROUTING_INTERFACE

No GPU run is admitted before G0 passes.
