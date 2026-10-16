from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "qwen_dinov3_hqnet_pipeline" / "configs"
BACKBONE = ROOT / "qwen_dinov3_hqnet_pipeline" / "mmdet_ext" / "dinov3_backbone.py"
LAUNCHER = ROOT / "qwen_dinov3_hqnet_pipeline" / "tools" / "train_stage2.py"
NATIVE_TRAIN = ROOT / "qwen_dinov3_hqnet_pipeline" / "tools" / "train_stage2_qwen_dino_hqnet_native.py"
ENV_CHECKER = ROOT / "qwen_dinov3_hqnet_pipeline" / "tools" / "check_stage2_env.py"
TRAIN_SMOKE = ROOT / "qwen_dinov3_hqnet_pipeline" / "tools" / "smoke_stage2_humanv2_train.py"
README = ROOT / "qwen_dinov3_hqnet_pipeline" / "README.md"
ALL_TASK_METRIC_DOUBLE = 'metric=["bbox", "attribute", "kpt", "segm"]'
ALL_TASK_METRIC_SINGLE = "metric=['bbox', 'attribute', 'kpt', 'segm']"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _has_all_task_metric(text: str) -> bool:
    return ALL_TASK_METRIC_DOUBLE in text or ALL_TASK_METRIC_SINGLE in text


class Stage2StaticTest(unittest.TestCase):
    def test_stage2_main_config_evaluates_all_2d_tasks(self):
        text = _read(CONFIG_DIR / "stage2_qwen_ie_dinov3_hqnet.py")
        self.assertTrue(_has_all_task_metric(text))

    def test_stage2_has_dino_only_baseline_config(self):
        baseline = CONFIG_DIR / "stage2_dinov3_hqnet_baseline.py"
        self.assertTrue(baseline.exists())
        text = _read(baseline)
        self.assertIn("DINOv3PyramidBackbone", text)
        self.assertNotIn("FrozenQwenDINOv3Backbone", text)
        self.assertTrue(_has_all_task_metric(text))

    def test_backbone_supports_feature_level_fusion_modes(self):
        text = _read(BACKBONE)
        tree = ast.parse(text)
        self.assertIn("feature_add", text)
        self.assertIn("feature_concat", text)
        self.assertIn("input_add", text)
        self.assertIn("none", text)
        self.assertTrue(any(isinstance(node, ast.ClassDef) and node.name == "FrozenQwenDINOv3Backbone" for node in tree.body))
        self.assertIn("_fuse_feature_maps", text)
        self.assertIn("qwen_feature_projs", text)

    def test_launcher_exposes_stage2_ablation_switches(self):
        text = _read(LAUNCHER)
        native_text = _read(NATIVE_TRAIN)
        self.assertIn("--backbone-mode", native_text)
        self.assertIn("stage2_dinov3_hqnet_baseline.py", native_text)
        self.assertIn("--fusion-mode", text)
        self.assertIn("--freeze-dino", text)
        self.assertIn("STAGE2_FUSION_MODE", text)
        self.assertIn("STAGE2_FREEZE_DINO", text)

    def test_env_checker_exists_for_stage2_runtime(self):
        self.assertTrue(ENV_CHECKER.exists())
        text = _read(ENV_CHECKER)
        for expected in ["mmcv", "mmdet", "transformers", "DINOV3_MODEL_NAME", "COCO_UNIHUMAN_TRAIN_JSON"]:
            self.assertIn(expected, text)

    def test_humanv2_stage2_train_smoke_exists(self):
        self.assertTrue(TRAIN_SMOKE.exists())
        text = _read(TRAIN_SMOKE)
        for expected in ["SimpleStage2Heads", "FrozenQwenDINOv3Backbone", "bbox_loss", "optimizer", "qwen_trainable_params"]:
            self.assertIn(expected, text)

    def test_readme_documents_baseline_and_feature_fusion(self):
        text = _read(README)
        self.assertIn("DINO-only baseline", text)
        self.assertIn("feature_add", text)
        self.assertIn("feature_concat", text)
        self.assertIn(ALL_TASK_METRIC_DOUBLE, text)


if __name__ == "__main__":
    unittest.main()
