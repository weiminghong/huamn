import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Sam2Model,
    Sam2Processor,
)

from .io_utils import parse_json_object_from_text
from .labels import LIP_LABELS, LIP_NAME_TO_ID, normalize_lip_label
from .prompting import build_qwen_parsing_prompt


@dataclass
class PredictedPart:
    class_id: int
    class_name: str
    bbox: List[float]
    score: float
    person_index: int


def _to_device(obj: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    for k, v in obj.items():
        if isinstance(v, torch.Tensor):
            obj[k] = v.to(device)
    return obj


def _clamp_box(box: List[float], width: int, height: int) -> Optional[List[float]]:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(0.0, min(x2, width - 1))
    y2 = max(0.0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


class QwenSam2ParsingPipeline:
    def __init__(
        self,
        qwen_model_name_or_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        sam2_model_name_or_path: str = "facebook/sam2.1-hiera-large",
        max_new_tokens: int = 768,
        qwen_dtype: str = "auto",
        sam2_dtype: str = "auto",
        device: Optional[str] = None,
        extra_prompt: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.max_new_tokens = max_new_tokens
        self.prompt = build_qwen_parsing_prompt(extra_prompt)

        qwen_torch_dtype = self._resolve_dtype(qwen_dtype)
        sam2_torch_dtype = self._resolve_dtype(sam2_dtype)

        self.qwen_processor = AutoProcessor.from_pretrained(
            qwen_model_name_or_path,
            trust_remote_code=True,
        )
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            qwen_model_name_or_path,
            torch_dtype=qwen_torch_dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.qwen_model.eval()

        self.sam2_processor = Sam2Processor.from_pretrained(sam2_model_name_or_path)
        self.sam2_model = Sam2Model.from_pretrained(
            sam2_model_name_or_path,
            torch_dtype=sam2_torch_dtype,
        ).to(self.device)
        self.sam2_model.eval()

    def _resolve_dtype(self, value: str):
        value = (value or "auto").lower()
        if value == "bf16":
            return torch.bfloat16
        if value == "fp16":
            return torch.float16
        if value == "fp32":
            return torch.float32
        return "auto"

    def _build_messages(self, image: Image.Image) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

    def generate_structured_parsing(self, image: Image.Image) -> Tuple[str, Dict[str, Any], bool]:
        messages = self._build_messages(image)
        inputs = self.qwen_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = _to_device(inputs, self.device)

        with torch.no_grad():
            output_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]
        text = self.qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        obj = parse_json_object_from_text(text)
        valid = obj is not None
        if obj is None:
            obj = {"persons": []}
        return text, obj, valid

    def extract_parts(self, struct: Dict[str, Any], image_size: Tuple[int, int]) -> List[PredictedPart]:
        width, height = image_size
        persons = struct.get("persons", [])
        if not isinstance(persons, list):
            return []

        parts: List[PredictedPart] = []
        for person_index, person in enumerate(persons):
            if not isinstance(person, dict):
                continue
            part_list = person.get("parts", [])
            if not isinstance(part_list, list):
                continue
            for item in part_list:
                if not isinstance(item, dict):
                    continue
                class_name = normalize_lip_label(item.get("label", ""))
                if class_name is None:
                    continue
                bbox = _clamp_box(item.get("bbox", []), width, height)
                if bbox is None:
                    continue
                score = float(item.get("score", 1.0)) if item.get("score", 1.0) is not None else 1.0
                parts.append(
                    PredictedPart(
                        class_id=LIP_NAME_TO_ID[class_name],
                        class_name=class_name,
                        bbox=bbox,
                        score=score,
                        person_index=person_index,
                    )
                )
        return parts

    def sam2_predict_masks(self, image: Image.Image, parts: List[PredictedPart]) -> List[Tuple[PredictedPart, np.ndarray, float]]:
        if not parts:
            return []

        input_boxes = [[part.bbox for part in parts]]
        inputs = self.sam2_processor(
            images=image,
            input_boxes=input_boxes,
            return_tensors="pt",
        )
        inputs = _to_device(inputs, self.device)

        with torch.no_grad():
            outputs = self.sam2_model(**inputs, multimask_output=False)

        masks_per_image = self.sam2_processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
        )[0]
        iou_scores = outputs.iou_scores.detach().float().cpu().view(-1).tolist()

        result = []
        for idx, part in enumerate(parts):
            mask = masks_per_image[idx]
            if torch.is_tensor(mask):
                mask = mask.numpy()
            mask = np.asarray(mask)
            if mask.ndim > 2:
                mask = np.squeeze(mask, axis=0)
            score = iou_scores[idx] if idx < len(iou_scores) else 1.0
            result.append((part, mask > 0, float(score)))
        return result

    def build_semantic_map(self, image: Image.Image, parts: List[PredictedPart]) -> np.ndarray:
        width, height = image.size
        semantic = np.zeros((height, width), dtype=np.int64)
        confidence = np.full((height, width), -1e9, dtype=np.float32)

        sam2_outputs = self.sam2_predict_masks(image, parts)
        for part, mask, mask_score in sorted(sam2_outputs, key=lambda x: x[2], reverse=True):
            if mask.shape != semantic.shape:
                mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.NEAREST)) > 0
            part_conf = float(mask_score * max(part.score, 1e-6))
            update = mask & (part_conf > confidence)
            semantic[update] = part.class_id
            confidence[update] = part_conf
        return semantic

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        pred_text, struct, valid_json = self.generate_structured_parsing(image)
        parts = self.extract_parts(struct, image.size)
        semantic_map = self.build_semantic_map(image, parts)
        return {
            "prediction_text": pred_text,
            "prediction_struct": struct,
            "valid_json": valid_json,
            "predicted_parts": [
                {
                    "label": p.class_name,
                    "bbox": [round(v, 2) for v in p.bbox],
                    "score": p.score,
                    "person_index": p.person_index,
                }
                for p in parts
            ],
            "semantic_map": semantic_map,
        }

    def semantic_map_to_png(self, semantic_map: np.ndarray, path: str) -> None:
        Image.fromarray(semantic_map.astype(np.uint8), mode="L").save(path)

    @property
    def class_names(self) -> List[str]:
        return list(LIP_LABELS)
