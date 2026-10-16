from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from .protocol import PacketRef, packet_tasks


class SamplingError(ValueError):
    pass


@dataclass(frozen=True)
class SamplingAudit:
    epoch_size: int
    seed: int
    task_counts: Dict[str, int]
    dataset_task_counts: Dict[str, int]


def build_packet_index(
    records: Sequence[Mapping], *, joint_min_tasks: int = 2
) -> Dict[str, Dict[str, List[PacketRef]]]:
    index: Dict[str, Dict[str, List[PacketRef]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record_index, record in enumerate(records):
        tasks = packet_tasks(record, allow_joint=False)
        for task in tasks:
            index[task][str(record["dataset"])].append(
                PacketRef(record_index, str(record["dataset"]), task, False)
            )
        if len(tasks) >= joint_min_tasks:
            index["all"][str(record["dataset"])].append(
                PacketRef(record_index, str(record["dataset"]), "all", True)
            )
    return {task: dict(groups) for task, groups in index.items()}


class BalancedPacketSampler:
    """Task-first sampling with capped dataset-size influence."""

    def __init__(
        self,
        index: Dict[str, Dict[str, List[PacketRef]]],
        task_weights: Mapping[str, float],
        *,
        epoch_size: int,
        seed: int,
    ) -> None:
        self.index = index
        self.task_weights = {
            task: float(weight)
            for task, weight in task_weights.items()
            if float(weight) > 0
        }
        self.epoch_size = int(epoch_size)
        self.seed = int(seed)
        if self.epoch_size <= 0:
            raise SamplingError("epoch_size must be positive")
        missing = set(self.task_weights) - set(index)
        if missing:
            raise SamplingError(
                f"weighted tasks have no eligible records: {missing}"
            )
        if not self.task_weights:
            raise SamplingError("at least one task weight is required")

    @staticmethod
    def _choice(
        rng: random.Random,
        items: Sequence[str],
        weights: Sequence[float],
    ) -> str:
        total = sum(weights)
        mark = rng.random() * total
        running = 0.0
        for item, weight in zip(items, weights):
            running += weight
            if mark <= running:
                return item
        return items[-1]

    def epoch(self, epoch: int) -> Tuple[List[PacketRef], SamplingAudit]:
        rng = random.Random(self.seed + int(epoch))
        tasks = sorted(self.task_weights)
        task_prob = [self.task_weights[task] for task in tasks]
        packets: List[PacketRef] = []
        task_counts: Dict[str, int] = defaultdict(int)
        dataset_task_counts: Dict[str, int] = defaultdict(int)
        for _ in range(self.epoch_size):
            task = self._choice(rng, tasks, task_prob)
            groups = self.index[task]
            datasets = sorted(groups)
            dataset_prob = [
                math.sqrt(len(groups[name])) for name in datasets
            ]
            dataset = self._choice(rng, datasets, dataset_prob)
            packet = groups[dataset][rng.randrange(len(groups[dataset]))]
            packets.append(packet)
            task_counts[task] += 1
            dataset_task_counts[f"{dataset}:{task}"] += 1
        return packets, SamplingAudit(
            epoch_size=self.epoch_size,
            seed=self.seed + int(epoch),
            task_counts=dict(sorted(task_counts.items())),
            dataset_task_counts=dict(sorted(dataset_task_counts.items())),
        )
