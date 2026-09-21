from dataclasses import dataclass

import numpy as np
import torch

from utils.aig_schema import GateType


@dataclass
class PackedDAGSchedule:
    indices: torch.Tensor
    counts: list

    def to(self, device):
        packed = self.indices.to(device, non_blocking=True)
        schedule = []
        offset = 0
        for na, nn in self.counts:
            and_nodes = packed[offset:offset + na]
            offset += na
            and_edges = packed[offset:offset + 4 * na].view(2, 2 * na)
            offset += 4 * na
            not_nodes = packed[offset:offset + nn]
            offset += nn
            not_edges = packed[offset:offset + 2 * nn].view(2, nn)
            offset += 2 * nn
            schedule.append((and_nodes, and_edges, not_nodes, not_edges))
        return schedule, len(self.counts) + 1


def prepare_cpu_schedule(gate, edge_index, forward_level):
    if gate.device.type != "cpu":
        raise ValueError("prepare_cpu_schedule requires CPU graph tensors")
    gate = gate.reshape(-1).numpy()
    levels = forward_level.reshape(-1).numpy()
    edges = edge_index.numpy()
    if edges.size == 0:
        return PackedDAGSchedule(torch.empty(0, dtype=torch.long), [])
    dst = edges[1]
    starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]])
    expected = np.count_nonzero((gate == int(GateType.AND)) | (gate == int(GateType.NOT)))
    if len(starts) != expected:
        edges = edges[:, np.argsort(dst, kind="stable")]
        dst = edges[1]
        starts = np.flatnonzero(np.r_[True, dst[1:] != dst[:-1]])
    nodes = dst[starts]
    num_levels = int(levels.max()) + 1
    groups = []
    for kind, arity in [(GateType.AND, 2), (GateType.NOT, 1)]:
        mask = gate[nodes] == int(kind)
        selected = nodes[mask]
        positions = starts[mask]
        order = np.lexsort((selected, levels[selected]))
        selected, positions = selected[order], positions[order]
        boundaries = np.r_[0, np.cumsum(np.bincount(levels[selected], minlength=num_levels))]
        groups.append((selected, positions, boundaries, arity))
    chunks, counts = [], []
    for level in range(1, num_levels):
        sizes = []
        for selected, positions, boundaries, arity in groups:
            begin, end = boundaries[level:level + 2]
            targets = selected[begin:end]
            starts_at_level = positions[begin:end]
            sources = edges[0, (starts_at_level[:, None] + np.arange(arity)).reshape(-1)]
            local_dst = np.repeat(np.arange(len(targets), dtype=np.int64), arity)
            chunks.extend((targets, sources, local_dst))
            sizes.append(len(targets))
        counts.append(tuple(sizes))
    packed = torch.from_numpy(np.concatenate(chunks).astype(np.int64, copy=False))
    return PackedDAGSchedule(packed, counts)
