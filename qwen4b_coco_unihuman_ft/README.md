# Qwen COCO-UniHuman Box/Pose/Attribute Fine-tuning

这个目录是独立微调工程，当前任务只保留：

- `box`
- `pose`
- `attribute`（`gender`, `age`）

不再包含 segmentation/parsing 训练目标。

## 1. 你的服务器路径（已适配）

- 训练标注：`/home/priv/wmh_project/coco_person/labels/coco_unihuman_train_v1.json`
- 验证标注：`/home/priv/wmh_project/coco_person/labels/coco_unihuman_val_v1.json`
- 图像目录：`/home/priv/wmh_project/coco_person/images`
- 图像名示例：`000000420612.jpg`

转换脚本兼容两种结构：
- COCO 标准：`images + annotations`
- UniHuman 常见：`images[].instance`（参考 `COCO-UniHuman-main/HumanQueryNet` 的读取习惯）

## 2. 先做数据转换（服务器执行）

```bash
python qwen4b_coco_unihuman_ft/convert_coco_unihuman.py \
  --train-ann-json /home/priv/wmh_project/coco_person/labels/coco_unihuman_train_v1.json \
  --val-ann-json /home/priv/wmh_project/coco_person/labels/coco_unihuman_val_v1.json \
  --image-root /home/priv/wmh_project/coco_person/images \
  --output-dir /home/priv/wmh_project/coco_person/ft_box_pose_attr_data \
  --emit-swift-format
```

输出文件：
- `train.jsonl` / `val.jsonl`（给 `train_sft.py`）
- `train_swift.jsonl` / `val_swift.jsonl`（给 `swift sft`）
- `summary.json`

GT `target_text` 格式：

```json
{
  "box": [{"person_id": 1, "bbox_xyxy": [x1, y1, x2, y2]}],
  "pose": [{"person_id": 1, "keypoints": [[x, y, v], "... 17 COCO keypoints ..."]}],
  "attribute": [{"person_id": 1, "gender": 0, "age": 25}]
}
```

其中 `person_id` 用来对齐同一个人的 box、pose 和属性；属性只在 COCO-UniHuman `valid.gender` / `valid.age` 有效时写入。

## 3. 用 ms-swift 微调（推荐）

你可以直接用：
- `qwen4b_coco_unihuman_ft/scripts/run_train_swift_lora.sh`

或命令行手动执行（示例）：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --dataset /home/priv/wmh_project/coco_person/ft_box_pose_attr_data/train_swift.jsonl \
  --val_dataset /home/priv/wmh_project/coco_person/ft_box_pose_attr_data/val_swift.jsonl \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 16 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --max_length 4096 \
  --output_dir /home/priv/wmh_project/coco_person/outputs/qwen_vl_box_pose_attr_lora \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4
```

## 4. 用本目录的 train_sft.py 微调（备选）

1. 修改配置：
- [qwen4b_box_pose_sft_server.yaml](D:\科研项目\人体感知\human\qwen4b_coco_unihuman_ft\configs\qwen4b_box_pose_sft_server.yaml)
2. 启动：

```bash
python qwen4b_coco_unihuman_ft/train_sft.py \
  --config qwen4b_coco_unihuman_ft/configs/qwen4b_box_pose_sft_server.yaml
```

## 5. Prompt 与损失

- Prompt 在 [prompts.py](D:\科研项目\人体感知\human\qwen4b_coco_unihuman_ft\prompts.py)
- Collator 在 [dataset.py](D:\科研项目\人体感知\human\qwen4b_coco_unihuman_ft\dataset.py)
- Loss 机制：`prompt + target_text`，prompt token 标注为 `-100`，只对答案 token 做 CE loss

## 6. 重要说明

如果你要喂图片，模型需要是 VL（如 Qwen2.5-VL 系列）。纯文本 Qwen-4B 不能直接吃图像。
