from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.formatting import to_tensor


@PIPELINES.register_module()
class LoadQwenTeacherFeature:
    """Load offline Stage1 teacher features from a sidecar index file."""

    def __init__(
        self,
        index_file: str,
        root_dir: str = "",
        output_key: str = "teacher_feature",
        strict: bool = False,
    ) -> None:
        self.index_file = str(index_file)
        self.root_dir = str(root_dir)
        self.output_key = str(output_key)
        self.strict = bool(strict)
        with open(self.index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping json in {self.index_file}")
        self.index = {str(k): str(v) for k, v in data.items()}

    def _resolve_path(self, results: Dict[str, Any]) -> Path | None:
        img_info = results.get("img_info", {})
        image_id = str(img_info.get("id", ""))
        file_name = str(img_info.get("filename", img_info.get("file_name", "")))
        rel_path = self.index.get(image_id) or self.index.get(file_name)
        if not rel_path:
            return None
        path = Path(rel_path)
        if self.root_dir and not path.is_absolute():
            path = Path(self.root_dir) / path
        return path

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        feature_path = self._resolve_path(results)
        if feature_path is None or not feature_path.exists():
            if self.strict:
                raise FileNotFoundError(f"Teacher feature missing for image: {results.get('img_info', {})}")
            results[self.output_key] = np.zeros((1, 1, 1), dtype=np.float32)
            return results

        payload = torch.load(str(feature_path), map_location="cpu")
        if isinstance(payload, dict):
            feature = payload.get("feature_map", payload.get("teacher_feature", None))
            if feature is None and "tensor" in payload:
                feature = payload["tensor"]
        else:
            feature = payload
        if torch.is_tensor(feature):
            feature = feature.float().cpu()
        else:
            feature = torch.tensor(feature, dtype=torch.float32)
        results[self.output_key] = feature
        return results


@PIPELINES.register_module()
class TeacherFeatureFormatBundle:
    """Wrap the teacher feature tensor so Collect can forward it to the model."""

    def __init__(self, key: str = "teacher_feature") -> None:
        self.key = str(key)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if self.key not in results:
            return results
        value = results[self.key]
        if torch.is_tensor(value):
            tensor = value
        else:
            tensor = to_tensor(value)
        results[self.key] = DC(tensor, stack=False, cpu_only=False)
        return results
