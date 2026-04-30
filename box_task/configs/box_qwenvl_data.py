CONFIG = {
    "model_name_or_path": "/home/priv/project/model/Qwen3.5-4B",
    "data": {
        "train": {
            "image_dir": "data/human/images/train",
            "label_dir": "data/human/box_labels/train",
            "list_file": None,
        },
        "val": {
            "image_dir": "data/human/images/val",
            "label_dir": "data/human/box_labels/val",
            "list_file": None,
        },
    },
    "output_dir": "Outputs/box_task/qwen_vl_box",
    "eval_batch_size": 1,
    "dataloader_num_workers": 2,
    "max_length": 4096,
    "max_new_tokens": 512,
    "eval_max_new_tokens": 512,
    "bf16": True,
    "fp16": False,
    "score_thr": 0.0,
    "nms_iou_thr": 0.5,
    "max_per_img": 100,
    "eval_iou_thr": 0.5,
    "extra_prompt": None,
}
