# Human Slot Memory Pipeline Design

This document describes the current `human_slot_memory_pipeline` implementation, its intended training flow, and the parts that are already runnable versus the parts that are still scaffolds.

## 中文总览

### 研究动机

我们想解决的核心问题是：**人体感知任务很多，但现有方法往往把检测、姿态、属性、分割、关系理解拆成彼此割裂的多个子系统**。这样做虽然在单任务上容易优化，但会带来几个明显问题：

1. **任务之间信息没有共享**
   - 人体框、关键点、性别、年龄这些信息本来是强相关的。
   - 例如，小框人体很难做姿态，但框本身仍然是可靠的；姿态缺失时，属性和检测信息仍然可能有价值。
   - 如果每个任务独立训练，很多跨任务约束就没有被利用起来。

2. **UniHuman 这类数据集天然是部分标注的**
   - 不是每个人都有完整的 `box + pose + attr + seg` 标注。
   - 如果用传统“所有任务都必须有标签”的训练范式，很容易把“缺标”误当成“负样本”，从而引入错误监督。

3. **VLM 有很强的开放式理解能力，但 dense backbone 更适合稳定的像素级/实例级预测**
   - Qwen 这类视觉语言模型擅长结构化理解、语言对齐和复杂规则输出。
   - DINOv3 这类视觉 backbone 更擅长稳定提取稠密视觉特征，适合作为下游检测/姿态 student。
   - 因此，一个自然的想法是：**先让 VLM 负责“教会系统怎么理解人”，再把这种人级结构蒸馏给 dense vision model，最后再把 dense prediction 回流给语言模型做高层推理。**

所以，这条 pipeline 的 motivation 可以概括成一句话：

> 用 VLM 学“人”的结构语义，用 DINO 学稳定 dense 表征，再把 dense human memory 回流给语言模型，形成一个统一的人体理解闭环。

### 整体路线

整个路线可以理解为“三步走 + 一个前置验证”：

1. **Stage-0：先验证 teacher prompt 是否稳定**
   - 不训练，只测试 Qwen-VL 在当前 prompt 下，能不能稳定输出干净的 XML 结构结果。
   - 目的不是追求最终最优精度，而是确认 teacher 的输出格式可靠，不带杂乱 thinking，不乱字段，不乱 schema。

2. **Stage-1：训练结构化人体 teacher**
   - 用 Qwen-VL 从图像直接生成“按人组织”的结构化结果。
   - 每个人对应一个 slot，slot 中包含当前关心的任务字段。
   - 这一阶段的本质是：**把开放式 VLM 压成一个可训练、可蒸馏、可对齐的人体结构教师。**

3. **Stage-2：把 teacher 的人级结构蒸馏到 DINOv3 dense student**
   - 用 DINOv3 提取 dense visual feature。
   - 再通过 query/router/head，把 dense 特征变成固定数量的人体 query。
   - 每个 query 预测 box、pose、attr 等结构化结果。
   - 这一阶段的本质是：**把“语言模型会说的人体结构”转成“视觉 backbone 会稳定预测的人体结构”。**

4. **Stage-3：把 dense human query 压缩成 memory token，再回流给语言模型**
   - 把 Stage-2 的 dense query 压缩成更紧凑的人体记忆表示。
   - 再让 Qwen 基于这些 memory token 做 caption、QA、relation reasoning 等高层任务。
   - 这一阶段的本质是：**不再让语言模型直接从原图重新“猜”，而是让它基于已经结构化的人体记忆来推理。**

### 为什么这样分阶段

这样分阶段不是为了把系统做复杂，而是为了把不同模型的优势分清楚：

- **VLM 擅长：**
  - 结构化理解
  - 长指令约束
  - 复杂语义对齐
  - 多任务统一输出

- **DINOv3 擅长：**
  - 稳定视觉特征
  - 稠密表示
  - 更容易接检测/姿态类 head
  - 推理时更像标准视觉模型，效率和稳定性更好

所以我们不希望：

- 让 VLM 一直承担所有 dense prediction
- 也不希望 DINO 从零自己学会复杂的人级语义组织

而是希望：

- **VLM 先学“结构语义”**
- **DINO 再学“稳定落地”**
- **memory loop 再把 dense 结果提升回可推理的语言空间**

### 当前项目里的真实落地范围

虽然 pipeline 的长期目标更大，但当前这版代码里，真正 active 的任务范围是：

- `box`
- `pose`
- `gender`
- `age`

也就是说，现在的主线是一个 **四任务统一的人体结构 pipeline**。

`segmentation`、`relation`、`caption`、`action` 这些字段在设计里保留了接口，但当前 runnable path 里还没有全部真正训起来。

### 当前每个 Stage 在做什么

#### Stage-0：Prompt 验证器

它做的事情是：

- 给 Qwen-VL 一个严格的 XML prompt
- 检查输出里有没有 `<think>`、重复 answer、格式错误
- 必要时做一次 repair
- 用统一评估脚本看 `box / pose / gender / age` 的结果

它的意义是：

- 保证 Stage-1 的 teacher 格式是干净的
- 避免后面训练数据和推理结果 schema 漂移

#### Stage-1：结构化 teacher

它做的事情是：

- 从原始 UniHuman 标注生成 SFT 用 JSONL
- 让 Qwen-VL 学会从图像直接输出 `<person> ... </person>` 结构
- 每个人一个 slot，slot 里放 box、pose、gender、age

它的意义是：

- 把原始人体标注统一成“人级结构表示”
- 给后面的 dense student 提供 teacher 监督来源

#### Stage-2：dense student

它要做的事情是：

- 用 DINOv3 提特征
- 用 slot router 产生固定数量的人体 query
- 用多个 head 分别预测 box、pose、attr
- 用 partial-label loss 处理缺标情况

它的意义是：

- 把 teacher 的结构能力蒸馏成一个更标准、更稳定的视觉模型
- 使后续推理不再完全依赖 VLM 的生成式输出

#### Stage-3：memory loop

它要做的事情是：

- 把 dense query 压缩成 human memory tokens
- 再让语言模型基于 memory 做高层输出

它的意义是：

- 让语言推理建立在结构化人体证据之上
- 而不是每次都直接从原图端到端生成

### 一句话总结

如果用一句中文概括这条路线：

> 这条 pipeline 的核心思想，就是先让 Qwen 学会“按人理解图像并输出结构化人体 slot”，再把这种 slot 级知识蒸馏给 DINOv3，最后把 dense human query 压缩成 memory token 回流给语言模型，形成一个统一的人体感知与推理闭环。

## 1. Goal

The pipeline is built around a simple idea:

- use a vision-language model as a **structured human teacher**
- convert each image into **person-level slots**
- distill those slots into a dense vision backbone
- compress dense predictions into **human memory tokens**
- feed those memory tokens back to a language model for grounded reasoning

At the current stage of the project, the active task scope is:

- `box`
- `pose`
- `gender`
- `age`

`segmentation` was discussed earlier, but it is currently not active in the runnable path.

## 2. High-Level Stages

The repository currently organizes the idea into four stages:

1. **Stage-0: Teacher prompt evaluation**
2. **Stage-1: Structured MLLM teacher SFT**
3. **Stage-2: Dense human student**
4. **Stage-3: Dense-to-language memory loop**

These stages share the same raw supervision source:

- `/home/priv/wmh_project/coco_person/labels/coco_unihuman_train_v1.json`
- `/home/priv/wmh_project/coco_person/labels/coco_unihuman_val_v1.json`

## 3. Data Representation

The core data structure is a `HumanSample` containing multiple `PersonSlot` instances:

- sample-level fields:
  - `sample_id`
  - `image_path`
  - `width`
  - `height`
- person-level fields:
  - `box_xyxy`
  - `pose17_xyv`
  - `attributes`
  - `task_mask`

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\schema.py:28`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\schema.py:56`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\schema.py:83`

`task_mask` is important because COCO-UniHuman is partially labeled. A person may have a valid box but no valid pose or no valid attributes. The mask allows each task to contribute loss only when the ground truth exists.

## 4. Stage-0: Teacher Prompt Evaluation

### 4.1 Purpose

Stage-0 is not training. It is a prompt-and-decoding evaluation stage used to test whether the Qwen-VL teacher can directly output clean structured person predictions.

### 4.2 Input / Output

Input:

- image
- strict XML prompt

Output:

- `<answer> ... </answer>` XML with repeated `<person>` blocks
- per-person tags:
  - `<box>`
  - `<pose>`
  - `<gender>`
  - `<age>`
  - `<score>`

Relevant prompt definitions:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\prompts.py:9`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\prompts.py:46`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\prompts.py:71`

### 4.3 Current behavior

The evaluation script:

- uses a strict XML prompt
- suppresses chain-of-thought / `<think>`
- truncates to the first `<answer>`
- optionally performs one repair pass if the raw output is malformed
- evaluates predictions against normalized UniHuman targets

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\eval_stage0_teacher.py:21`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\eval_stage0_teacher.py:77`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\eval_stage0_teacher.py:180`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\eval_stage0_teacher.py:241`

### 4.4 Role in the whole pipeline

Stage-0 is mainly for:

- prompt debugging
- decoding stability checks
- early teacher quality estimation

It is not part of the final trainable loop, but it is useful to validate that the teacher format is stable before Stage-1 fine-tuning.

## 5. Stage-1: Structured MLLM Teacher

### 5.1 Purpose

Stage-1 fine-tunes a Qwen-VL model so that it outputs structured human slots directly from images.

This is the first trainable stage in the pipeline.

### 5.2 Current tasks

The active Stage-1 target contains:

- `box`
- `pose`
- `gender`
- `age`

The target text is generated from raw UniHuman annotations and formatted into XML-like blocks:

- `<person>`
- `<box>`
- `<pose>`
- `<gender>`
- `<age>`
- `<score>`

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\stage1_data.py:108`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\stage1_data.py:161`

### 5.3 Data preparation

Stage-1 does not directly read the raw COCO-UniHuman JSON inside the trainer. Instead, it first materializes an intermediate JSONL dataset:

- `stage1_sft_data/train.jsonl`
- `stage1_sft_data/val.jsonl`

The preparation step:

1. reads raw `coco_unihuman_*_v1.json`
2. keeps only valid labels for each task
3. converts each image into a structured training record
4. writes JSONL
5. generates a legacy SFT YAML for the actual trainer

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\train_stage1_teacher.py:20`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\train_stage1_teacher.py:42`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\train_stage1_teacher.py:63`

### 5.4 Training backend

The Stage-1 entry script is a wrapper. The actual fine-tuning is delegated to:

- `qwen4b_coco_unihuman_ft.train_sft`

That means Stage-1 currently reuses the mature SFT training code from the older Qwen fine-tuning directory rather than implementing a brand-new trainer inside `human_slot_memory_pipeline`.

Launch chain:

1. `human_slot_memory_pipeline.tools.train_stage1_teacher`
2. prepare JSONL + YAML
3. call `python -m qwen4b_coco_unihuman_ft.train_sft`

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\train_stage1_teacher.py:90`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\train_stage1_teacher.py:95`

### 5.5 Current Stage-1 config

Important current settings:

- model: `Qwen3.5-4B`
- `max_length=2048`
- `batch_size=1`
- `gradient_accumulation_steps=16`
- `gradient_checkpointing=true`
- `bf16=true`
- LoRA enabled on `q_proj`, `k_proj`, `v_proj`, `o_proj`
- vision tower frozen

Config file:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\configs\stage1_teacher.yaml:1`

### 5.6 Current status

Stage-1 is the most complete part of the trainable pipeline:

- data preparation works
- fine-tune launch works
- SwanLab logging works
- stage-wise evaluation hooks exist

## 6. Stage-2: Dense Human Student

### 6.1 Purpose

Stage-2 is intended to distill person-level teacher structure into a dense vision model based on DINOv3.

The idea is:

- DINOv3 extracts dense visual features
- a slot router turns dense features into a fixed set of person queries
- task heads decode those queries into structured outputs

### 6.2 Current architecture

The current student contains:

- `DinoV3Backbone`
- `HumanSlotRouter`
- `BoxHead`
- `PoseHead`
- `AttributeHead`
- placeholder `RelationHead`

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\dense_student.py:11`

Current active outputs:

- `queries`
- `validity_logits`
- `box_xyxy_norm`
- `pose17_xyv_norm`
- `gender_logits`
- `age`
- `relation_logits`

### 6.3 Current loss

The partial-label loss currently supervises:

- box L1 loss
- pose L1 loss
- validity BCE loss

The partial-label target builder uses task masks to avoid penalizing missing labels.

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\losses\partial_label.py:10`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\losses\partial_label.py:40`

Note:

- attribute loss is not yet fully wired into `PartialLabelLoss`
- relation is still a placeholder head
- teacher slot cache supervision is planned in config but not yet implemented in the current launcher

### 6.4 DINOv3 dependency

The current backbone loader expects a HuggingFace/transformers-style model directory and uses:

- `AutoModel.from_pretrained(..., trust_remote_code=True)`

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\dinov3_backbone.py:31`

This means the currently runnable path expects a DINOv3 model directory rather than a raw `.pth` checkpoint file.

### 6.5 Current status

Stage-2 is **architecture-ready but trainer-incomplete**.

The model, losses, and config scaffolding exist, but the entry point is still a scaffold:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\train_stage2_dense_student.py:1`

It currently prints config information and prepares the output directory, but it does not yet run a real dense student training loop.

## 7. Stage-3: Dense-to-Language Memory Loop

### 7.1 Purpose

Stage-3 is meant to feed dense human predictions back into a language model through compact memory tokens.

Conceptually:

1. Stage-2 produces dense person queries
2. queries are projected into memory tokens
3. memory tokens condition a Qwen model
4. Qwen produces grounded human-centric text

### 7.2 Current memory representation

The current memory helper summarizes each person slot into text-like fields:

- `<box>`
- `<pose_visible>`
- `<gender>`
- `<age>`

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\human_memory.py:10`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\human_memory.py:27`

The Stage-3 prompt currently asks for:

- grounded answers using box, pose, and attributes

Relevant code:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\prompts.py:139`

### 7.3 Current status

Stage-3 also exists mainly as a scaffold:

- `HumanMemoryProjector` exists
- `MemoryLanguageReasoner` exists
- config exists
- entry script exists

But there is not yet a full end-to-end training loop that jointly optimizes memory-conditioned reasoning on top of a trained Stage-2 student.

Relevant files:

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\human_memory.py:8`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\stage3_reasoner.py:8`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\train_stage3_memory_loop.py:1`

## 8. What Is Runnable Right Now

### Fully runnable

1. **Stage-0 prompt evaluation**
2. **Stage-1 data preparation**
3. **Stage-1 teacher fine-tuning**
4. **Stage-1 logging and stage-eval artifacts**
5. **dry run wiring check**

### Partially runnable / scaffold

1. **Stage-2 dense student architecture**
2. **Stage-2 losses**
3. **Stage-3 memory projection**
4. **Stage-3 language reasoner wrapper**

### Not yet complete

1. true Stage-2 training loop
2. teacher-slot cache distillation loop
3. full Stage-3 memory-conditioned training loop
4. stronger relation/action/caption tasks
5. segmentation branch reintroduction

## 9. Current Task Scope

The currently active task set in the implemented path is:

- `box`
- `pose`
- `gender`
- `age`

Concretely:

- Stage-0 evaluates these four tasks
- Stage-1 trains these four tasks
- Stage-2 architecture currently outputs box/pose/attr-related quantities
- Stage-3 memory summary currently keeps box/pose/attr information

`segmentation` is still present as a field in some general-purpose schema definitions, but it is not part of the current active training path.

## 10. Key Files

### Stage-0

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\eval_stage0_teacher.py`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\prompts.py`

### Stage-1

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\train_stage1_teacher.py`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\stage1_data.py`
- `D:\科研项目\人体感知\human\qwen4b_coco_unihuman_ft\train_sft.py`

### Stage-2

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\dense_student.py`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\dinov3_backbone.py`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\losses\partial_label.py`

### Stage-3

- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\human_memory.py`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\human_slot_memory\models\stage3_reasoner.py`
- `D:\科研项目\人体感知\human\human_slot_memory_pipeline\tools\train_stage3_memory_loop.py`

## 11. Recommended Reading Order

If a new collaborator wants to understand the project quickly, the most useful order is:

1. `README.md`
2. `PIPELINE.md`
3. `human_slot_memory/prompts.py`
4. `human_slot_memory/stage1_data.py`
5. `tools/train_stage1_teacher.py`
6. `human_slot_memory/models/dense_student.py`
7. `human_slot_memory/models/human_memory.py`

## 12. Short Summary

In one sentence:

> we first teach Qwen-VL to output person-centric structured supervision, then plan to distill that structure into DINOv3 dense queries, and finally feed the dense human memory back into a language model for grounded reasoning.

In the current implementation state:

- Stage-1 is the main working training path
- Stage-0 is the main evaluation/debugging path
- Stage-2 and Stage-3 already define the intended architecture, but still need full training loops to become complete
