from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PipelineConfig:
    raw: Dict[str, Any] = field(default_factory=dict)
    project: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, Any] = field(default_factory=dict)
    stage1: Dict[str, Any] = field(default_factory=dict)
    stage2: Dict[str, Any] = field(default_factory=dict)
    stage3: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)

    def output_dir(self) -> Path:
        return Path(str(self.project.get("output_dir", "outputs/human_slot_memory")))

    def ensure_output_dir(self) -> Path:
        path = self.output_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_config(path: str | Path) -> PipelineConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return PipelineConfig(
        raw=dict(raw),
        project=dict(raw.get("project", {})),
        paths=dict(raw.get("paths", {})),
        stage1=dict(raw.get("stage1", {})),
        stage2=dict(raw.get("stage2", {})),
        stage3=dict(raw.get("stage3", {})),
        optimization=dict(raw.get("optimization", {})),
        runtime=dict(raw.get("runtime", {})),
    )
