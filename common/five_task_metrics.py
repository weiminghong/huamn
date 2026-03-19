from typing import Any, Dict, List, Set

from .stage1_metrics import evaluate_stage1_records


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _to_key_set(items: Any) -> Set[str]:
    if not isinstance(items, list):
        return set()
    keys: Set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        kv = [f"{k}={item[k]}" for k in sorted(item.keys())]
        keys.add("|".join(kv))
    return keys


def evaluate_five_task_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    base_metrics = evaluate_stage1_records(records)
    if not records:
        return base_metrics

    attr_tp = attr_fp = attr_fn = 0
    rel_tp = rel_fp = rel_fn = 0
    seg_count_err = 0.0

    for rec in records:
        gt = rec.get("gt_struct", {}) or {}
        pred = rec.get("pred_struct", {}) or {}

        gt_attr = _to_key_set(gt.get("attribute", []))
        pred_attr = _to_key_set(pred.get("attribute", []))
        attr_tp += len(gt_attr & pred_attr)
        attr_fp += len(pred_attr - gt_attr)
        attr_fn += len(gt_attr - pred_attr)

        gt_rel = _to_key_set(gt.get("relation", []))
        pred_rel = _to_key_set(pred.get("relation", []))
        rel_tp += len(gt_rel & pred_rel)
        rel_fp += len(pred_rel - gt_rel)
        rel_fn += len(gt_rel - pred_rel)

        gt_seg = gt.get("parsing", [])
        pred_seg = pred.get("parsing", [])
        gt_count = len(gt_seg) if isinstance(gt_seg, list) else 0
        pred_count = len(pred_seg) if isinstance(pred_seg, list) else 0
        seg_count_err += abs(gt_count - pred_count)

    attr_p = _safe_div(attr_tp, attr_tp + attr_fp)
    attr_r = _safe_div(attr_tp, attr_tp + attr_fn)
    attr_f1 = _safe_div(2 * attr_p * attr_r, attr_p + attr_r)

    rel_p = _safe_div(rel_tp, rel_tp + rel_fp)
    rel_r = _safe_div(rel_tp, rel_tp + rel_fn)
    rel_f1 = _safe_div(2 * rel_p * rel_r, rel_p + rel_r)

    base_metrics.update(
        {
            "attribute_f1": attr_f1,
            "relation_f1": rel_f1,
            "parsing_seg_count_mae": seg_count_err / len(records),
        }
    )
    return base_metrics
