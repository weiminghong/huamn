from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from qwen_dinov3_hqnet_pipeline.common.qwen_utils import (
    dump_json,
    extract_hidden_size,
    extract_patch_size,
    extract_qwen_vision_features,
    load_qwen_vl_model,
    prepare_processor_image_inputs,
    resolve_vision_module,
    token_sequence_to_feature_map,
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage1 vision weights and offline teacher features.")
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--merge-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-merged-model", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    out_dir = Path(args.output_dir)
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_qwen_vl_model(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        torch_dtype=args.dtype,
        device=args.device,
        merge_lora=bool(args.merge_lora),
    )
    vision_module, vision_name = resolve_vision_module(model)
    patch_size = extract_patch_size(vision_module)
    hidden_size = extract_hidden_size(vision_module)

    if bool(args.save_merged_model):
        merged_dir = out_dir / "merged_stage1_model"
        merged_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(merged_dir))
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(merged_dir))

    torch.save(vision_module.state_dict(), out_dir / "vision_tower.pt")

    records = load_jsonl(args.input_jsonl)
    feature_index: Dict[str, str] = {}
    with torch.no_grad():
        for rec in records:
            image_path = Path(str(rec.get("image_abs", rec.get("image", ""))))
            if not image_path.is_absolute():
                image_path = Path(str(rec.get("image", "")))
            image = Image.open(image_path).convert("RGB")
            batch = prepare_processor_image_inputs(processor, image)
            batch = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in batch.items()}
            tokens = extract_qwen_vision_features(
                model,
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch.get("image_grid_thw", None),
            )
            fmap = token_sequence_to_feature_map(tokens, image_size=(image.height, image.width), patch_size=patch_size)
            payload = {
                "feature_map": fmap[0].detach().cpu().float(),
                "image_size": [image.width, image.height],
                "patch_size": int(patch_size),
                "hidden_size": int(hidden_size) if hidden_size is not None else None,
            }
            feat_path = feat_dir / f"{rec['id']}.pt"
            torch.save(payload, feat_path)
            feature_index[str(rec["id"])] = str(feat_path.relative_to(out_dir))

    teacher_manifest = {
        "model_name_or_path": str(args.model_name_or_path),
        "adapter_path": str(args.adapter_path),
        "vision_module_name": str(vision_name),
        "patch_size": int(patch_size),
        "hidden_size": int(hidden_size) if hidden_size is not None else None,
        "num_features": len(feature_index),
        "feature_index_file": str(out_dir / "teacher_index.json"),
    }
    dump_json(str(out_dir / "teacher_index.json"), feature_index)
    dump_json(str(out_dir / "teacher_manifest.json"), teacher_manifest)
    print(json.dumps(teacher_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
