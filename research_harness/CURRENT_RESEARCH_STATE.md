# Current Human Slot Research State

Updated: 2026-08-30

## Research Question

Can person-specific LISA-style pose and segment routing teach Qwen human-centric visual and decoder representations that progressively benefit DINOv3?

## Locked Story

1. Stage-1: Qwen text decoding produces short Human Slots, boxes, attributes, <POSE>, and <SEG>; routing-token states drive mature decoders over Qwen Vision features.
2. Stage-2: freeze the human-adapted Qwen visual path and transfer it into DINOv3.
3. Stage-3: add frozen person-specific <POSE>/<SEG>/slot LLM states beyond visual transfer.

## Stage-1 Dataflow

Image -> Qwen Vision -> multi-scale dense features + merger image tokens

Image tokens + prompt -> Qwen LLM -> per-person text slots

- Box, Gender, Age: native tied LM head.
- <SEG> state: projector -> SAM-style mask decoder over Qwen Vision features.
- <POSE> state: projector/FiLM -> box-conditioned SimCC decoder over Qwen Vision ROI features.
- The surrounding text slot and generated box bind both routing tokens to one human_id.

## Critical Visual-Learning Constraint

Qwen Vision is the only image encoder in the mainline. A SAM image encoder and ViTPose/RTMPose image encoder are upper-bound controls only. Dense gradients, frozen probes, detach controls, and image shuffles must prove that Qwen Vision itself learned human masks and keypoints.

## Training

1. Decoder warmup: freeze Qwen and train dense adapters, routing projectors, mask decoder, and pose decoder with GT boxes for 2,000 steps.
2. Joint tuning: train Vision LoRA, merger, LLM LoRA, token rows, adapters, and decoders with GT/jittered boxes for two epochs.
3. Generated-box finish: use generated boxes and finish adapters/decoders for one epoch.

## Active Mainline

E_STAGE1_QWEN_NATIVE_LISA_POSE_ROUTING_V1

Status: user_intent_aligned_lisa_pose_mainline_g0_pending

Current stage: Stage-1 Qwen-native LISA and pose-routing interface design

## Gates

- G0: routing ownership, decoder shapes, dense gradients, and no-specialist-encoder shortcut.
- G1: generated-box 64-image overfit across all five tasks.
- G2: separate 5K mask/pose pilots, joint 5K pilot, and frozen visual probes.
- G3 minimum: Box 52, Segment 50, Pose 60, Gender 50, Age 44 AP.

## Non-Contributions

SAM-style mask decoding and SimCC pose decoding are reused components. The contribution remains Qwen human adaptation and its Stage-2/Stage-3 transfer.

## Next Action

A005_FREEZE_LISA_POSE_ROUTING_INTERFACE

Freeze slot grammar, owner extraction, Qwen feature taps, decoder inputs, generated-box schedule, visual probes, and the 64-image manifest before implementation.
