from __future__ import annotations

import math
import torch
import os
import sys
from pathlib import Path
from torch_geometric.data import Batch, Dataset, Data


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.aig_schema import GateType
from model.deepgate2_schedule import prepare_cpu_schedule
from utils.structural_pe import (
    ALL_STATIC_FEATURES,
    EDGE_MAJOR_FEATURES,
    NODE_ID_FEATURES,
    TOKEN_BYTES,
    build_feature_config,
    build_simulation_classes,
)


_NODE_ID_KEYS = frozenset(NODE_ID_FEATURES)
_EDGE_MAJOR_KEYS = frozenset(EDGE_MAJOR_FEATURES)
_ZERO_INC_KEYS = frozenset(
    name for name in ALL_STATIC_FEATURES if name not in _NODE_ID_KEYS
) | {'num_base_nodes', 'y_known'}
_CAT_DIM_ZERO_KEYS = _ZERO_INC_KEYS | _NODE_ID_KEYS | {'y_index', 'y_sign'}

_SINGLE_CHANNEL_BATCH_FIELDS = (
    "gate",
    "edge_index",
    "forward_level",
    "forward_index",
    "functional_signature",
    "sim_class",
    "sim_class_sign",
    "y_index",
    "y_sign",
    "y_known",
    "circuit_name",
    "num_base_nodes",
)

class AIGData(Data):
    # Node-index fields are shifted when graphs are batched; the rest are not.
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'y_index':
            return self.num_nodes

        if key in {'y_sign', 'y_known'}:
            return 0

        if key in _NODE_ID_KEYS:
            return self.num_nodes
        if key in _ZERO_INC_KEYS:
            return 0

        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in _EDGE_MAJOR_KEYS:
            return 1
        if key in _CAT_DIM_ZERO_KEYS:
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

class DeepGate2Collater:
    def __init__(self, exclude_keys):
        self.exclude_keys = exclude_keys

    def __call__(self, graphs):
        batch = Batch.from_data_list(graphs, exclude_keys=self.exclude_keys)
        batch._deepgate2_cpu_schedule = prepare_cpu_schedule(
            batch.gate, batch.edge_index, batch.forward_level,
        )
        return batch


def validate_feature_config(feature_config: dict, source: str = "<dataset>") -> dict:
    if not isinstance(feature_config, dict):
        raise TypeError(f"[{source}] feature_config must be a dictionary")
    required = {
        "token_bytes",
        "simulation_bits",
        "feature_names",
    }
    missing = sorted(required - set(feature_config))
    extra = sorted(set(feature_config) - required)
    if missing or extra:
        raise ValueError(
            f"[{source}] invalid feature_config keys: missing={missing}, extra={extra}"
        )
    expected = build_feature_config()
    if feature_config != expected:
        raise ValueError(
            f"[{source}] feature_config does not match the current code contract"
        )
    return dict(expected)


def _validate_simulation_classes(data, source: str) -> None:
    expected_class, expected_sign = build_simulation_classes(
        data.functional_signature.cpu()
    )
    actual_class = data.sim_class.reshape(-1).cpu().to(torch.long)
    actual_sign = data.sim_class_sign.reshape(-1).cpu().to(torch.long)
    if not torch.equal(expected_class, actual_class):
        raise ValueError(
            f"[{source}] sim_class does not match functional_signature"
        )
    if not torch.equal(expected_sign, actual_sign):
        raise ValueError(
            f"[{source}] sim_class_sign does not match functional_signature"
        )
    if actual_class.numel():
        representatives = torch.unique(actual_class)
        if not torch.equal(
            actual_class.index_select(0, representatives), representatives
        ) or not bool(
            (actual_sign.index_select(0, representatives) == 1).all()
        ):
            raise ValueError(
                f"[{source}] simulation representatives must be fixed +1 nodes"
            )


def _validate_const_schema(
    gate: torch.Tensor,
    edge_index: torch.Tensor,
    batch: torch.Tensor | None,
) -> torch.Tensor:
    if gate.numel() == 0:
        raise ValueError("SECL requires a non-empty graph")
    batch = (
        torch.zeros_like(gate)
        if batch is None
        else batch.to(gate.device, dtype=torch.long).reshape(-1)
    )
    if batch.numel() != gate.numel() or int(batch[0]) != 0 or bool(
        (batch[1:] < batch[:-1]).any()
    ):
        raise ValueError("batch IDs must be contiguous and node aligned")
    graph_count = int(batch.max()) + 1
    if not torch.equal(
        torch.unique_consecutive(batch),
        torch.arange(graph_count, device=batch.device),
    ):
        raise ValueError("batch IDs must start at zero without gaps")
    node_count = torch.bincount(batch, minlength=graph_count)
    graph_start = torch.cumsum(node_count, dim=0) - node_count
    const_mask = gate == GateType.CONST
    const_indices = torch.where(const_mask)[0]
    const_count = torch.bincount(
        batch.index_select(0, const_indices), minlength=graph_count
    )
    if not bool((const_count == 1).all()) or not bool(
        (gate.index_select(0, graph_start) == GateType.CONST).all()
    ):
        raise ValueError("each graph must start with exactly one CONST0")
    if edge_index.numel() and bool(
        const_mask.index_select(0, edge_index[1]).any()
    ):
        raise ValueError("CONST0 must have no fanins")
    return const_indices

def _build_fanin_counts(
    gate: torch.Tensor, edge_index: torch.Tensor, validate: bool
) -> torch.Tensor:
    counts = torch.bincount(edge_index[1], minlength=gate.numel())
    if validate:
        expected = torch.zeros_like(gate)
        expected[gate == GateType.NOT] = 1
        expected[gate == GateType.AND] = 2
        supported = torch.zeros_like(gate, dtype=torch.bool)
        for gate_type in GateType:
            supported |= gate == gate_type
        if not bool(supported.all()) or not torch.equal(counts, expected):
            raise ValueError("gate types or AIG fanin counts are invalid")
    return counts

def validate_aig_graph(
    gate: torch.Tensor,
    edge_index: torch.Tensor,
    forward_level: torch.Tensor,
    forward_index: torch.Tensor,
    batch: torch.Tensor | None,
) -> None:
    num_nodes = gate.numel()
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise ValueError("edge_index must use int64 node IDs")
    if forward_level.shape != (num_nodes,) or forward_level.dtype != torch.long:
        raise ValueError("forward_level must be int64 [N]")
    if forward_index.shape != (num_nodes,) or forward_index.dtype != torch.long:
        raise ValueError("forward_index must be int64 [N]")
    if not torch.equal(
        torch.sort(forward_index).values,
        torch.arange(num_nodes, device=forward_index.device),
    ):
        raise ValueError("forward_index must be a permutation of all nodes")
    if edge_index.numel():
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            raise ValueError("edge_index contains an out-of-range node")
        src, dst = edge_index
        if not bool(
            (
                forward_level.index_select(0, src)
                < forward_level.index_select(0, dst)
            ).all()
        ):
            raise ValueError("every AIG edge must advance the forward level")
        if batch is not None:
            batch_flat = batch.reshape(-1)
            if not torch.equal(
                batch_flat.index_select(0, src),
                batch_flat.index_select(0, dst),
            ):
                raise ValueError("AIG edges must not cross graph boundaries")
    _build_fanin_counts(gate, edge_index, validate=True)
    _validate_const_schema(gate, edge_index, batch)

def validate_static_schema(
    data_list,
    feature_config: dict,
    source: str = "<dataset>",
    limit: int = None,
) -> int:
    validate_feature_config(feature_config, source=source)

    graphs = data_list if limit is None else data_list[:limit]
    allowed_fields = {
        "gate",
        "edge_index",
        "forward_level",
        "forward_index",
        "num_nodes",
        "num_base_nodes",
        "circuit_name",
        "y_index",
        "y_sign",
        "y_known",
        *ALL_STATIC_FEATURES,
    }
    for position, data in enumerate(graphs):
        name = getattr(data, "circuit_name", position)
        if not isinstance(data, AIGData):
            raise TypeError(
                f"[{source}] graph {name!r} must use AIGData so sim_class "
                "and y_index are offset safely during batching"
            )
        gate = data.gate.reshape(-1)
        num_nodes = int(gate.numel())
        try:
            extra_fields = sorted(set(data.keys()) - allowed_fields)
            missing_fields = sorted(allowed_fields - set(data.keys()))
            if extra_fields or missing_fields:
                raise ValueError(
                    "strict graph fields differ: "
                    f"missing={missing_fields}, extra={extra_fields}"
                )
            validate_aig_graph(
                gate,
                data.edge_index,
                data.forward_level,
                data.forward_index,
                None,
            )
            labels = getattr(data, "y_index", None)
            signs = getattr(data, "y_sign", None)
            known = getattr(data, "y_known", None)
            if (
                labels is None
                or labels.shape != (num_nodes,)
                or labels.dtype != torch.long
                or signs is None
                or signs.shape != (num_nodes,)
                or signs.dtype != torch.long
            ):
                raise ValueError("y_index/y_sign must be int64 [N]")
            if (
                known is None
                or known.shape != (num_nodes,)
                or known.dtype != torch.bool
            ):
                raise ValueError("y_known must be bool [N]")
            if bool(((labels < 0) | (labels >= num_nodes)).any()):
                raise ValueError("y_index contains an out-of-range representative")
            if bool(((signs != -1) & (signs != 1)).any()):
                raise ValueError("y_sign must contain only -1 or +1")
            representatives = torch.unique(labels)
            if not torch.equal(
                labels.index_select(0, representatives), representatives
            ) or not bool(
                (signs.index_select(0, representatives) == 1).all()
            ):
                raise ValueError(
                    "y_index representatives must be self-labelled with y_sign +1"
                )
            sim_class = getattr(data, "sim_class", None)
            if (
                sim_class is None
                or sim_class.shape != (num_nodes,)
                or sim_class.dtype != torch.long
            ):
                raise ValueError("sim_class must be int64 [N]")
            sim_sign = getattr(data, "sim_class_sign", None)
            if (
                sim_sign is None
                or sim_sign.shape != (num_nodes,)
                or sim_sign.dtype != torch.long
                or bool(((sim_sign != -1) & (sim_sign != 1)).any())
            ):
                raise ValueError("sim_class_sign must contain int64 ±1 [N]")
            _validate_simulation_classes(data, f"{source}:{name}")

            candidate = (gate == GateType.CONST) | (gate == GateType.AND)
            candidate_labels = labels[candidate]
            candidate_sim_class = sim_class[candidate]
            _, label_inverse, label_counts = torch.unique(
                candidate_labels, return_inverse=True, return_counts=True
            )
            group_count = label_counts.numel()
            maximum = torch.iinfo(torch.long).max
            device = candidate_labels.device
            base_min = torch.full(
                (group_count,), maximum, dtype=torch.long, device=device
            )
            base_max = torch.full(
                (group_count,), -1, dtype=torch.long, device=device
            )
            base_min.scatter_reduce_(
                0, label_inverse, candidate_sim_class, reduce="amin"
            )
            base_max.scatter_reduce_(
                0, label_inverse, candidate_sim_class, reduce="amax"
            )
            proved = label_counts > 1
            if bool((proved & (base_min != base_max)).any()):
                raise ValueError(
                    "a proved candidate class crosses a simulation class"
                )
            candidate_indices = torch.where(candidate)[0]
            proved_members = label_counts.index_select(0, label_inverse) > 1
            if bool(proved_members.any()):
                member_indices = candidate_indices[proved_members]
                representative_indices = labels.index_select(
                    0, member_indices
                )
                if not bool(candidate.index_select(
                    0, representative_indices
                ).all()):
                    raise ValueError(
                        "proved deployment classes require CONST/AND "
                        "representatives"
                    )
                exact_relation = signs.index_select(
                    0, member_indices
                ) * signs.index_select(0, representative_indices)
                simulation_relation = sim_sign.index_select(
                    0, member_indices
                ) * sim_sign.index_select(0, representative_indices)
                if not torch.equal(exact_relation, simulation_relation):
                    raise ValueError(
                        "proved equivalence polarity conflicts with "
                        "sim_class_sign"
                    )
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"[{source}] graph {name!r} failed schema validation: {exc}") from exc
    return len(graphs)


def estimate_graph_stats(
    data,
) -> dict:
    gate = data.gate.reshape(-1)
    known = data.y_known.reshape(-1).to(torch.bool)
    candidate = torch.where(
        ((gate == GateType.CONST) | (gate == GateType.AND)) & known
    )[0]
    candidate_count = int(candidate.numel())
    positive_nodes = 0
    proved_pairs = 0
    deployment_candidate_pairs = 0
    zero_bucket_candidates = 0
    stage2_anchor_reprs = 0
    pair_elements = 0
    stage2_pair_elements = 0
    if candidate_count:
        buckets = data.sim_class.index_select(0, candidate).to(torch.long)
        labels = data.y_index.index_select(0, candidate).to(torch.long)
        zero_class = int(data.sim_class[0])
        unique_buckets, bucket_sizes = torch.unique(
            buckets, sorted=True, return_counts=True
        )
        deployment_candidate_pairs = int(
            (bucket_sizes * (bucket_sizes - 1) // 2).sum()
        )
        multi_bucket = bucket_sizes > 1
        stage2_pair_elements = int(
            bucket_sizes[multi_bucket].square().sum()
        )
        zero_position = torch.searchsorted(
            unique_buckets,
            torch.tensor(zero_class, dtype=torch.long, device=buckets.device),
        )
        if (
            int(zero_position) < unique_buckets.numel()
            and int(unique_buckets[zero_position]) == zero_class
        ):
            zero_bucket_candidates = int(bucket_sizes[zero_position])

        node_count = int(gate.numel())
        group_keys = buckets * node_count + labels
        unique_groups, group_sizes = torch.unique(
            group_keys, sorted=True, return_counts=True
        )
        positive_group = group_sizes > 1
        positive_nodes = int(group_sizes[positive_group].sum())
        proved_pairs = int(
            (group_sizes * (group_sizes - 1) // 2).sum()
        )
        positive_group_buckets = unique_groups[positive_group] // node_count
        stage2_anchor_reprs = int(positive_group_buckets.numel())
        positive_representatives_per_bucket = torch.bincount(
            torch.searchsorted(unique_buckets, positive_group_buckets),
            minlength=unique_buckets.numel(),
        )
        pair_elements = int(
            (positive_representatives_per_bucket * bucket_sizes).sum()
        )

    depth = int(data.forward_level.max()) if data.forward_level.numel() else 0
    return {
        "circuit_name": str(data.circuit_name),
        "num_nodes": int(gate.numel()),
        "num_candidate_nodes": candidate_count,
        "num_positive_nodes": positive_nodes,
        "num_stage2_anchor_reprs": stage2_anchor_reprs,
        "num_proved_pairs": proved_pairs,
        "num_deployment_candidate_pairs": deployment_candidate_pairs,
        "num_zero_bucket_candidates": zero_bucket_candidates,
        "forward_depth": depth,
        "estimated_pair_elements": int(pair_elements),
        "estimated_stage2_pair_elements": int(stage2_pair_elements),
        "estimated_training_cost": int(
            depth
            + math.ceil(gate.numel() / 64)
            + math.ceil(stage2_pair_elements / 1024)
        ),
    }


class PackedSplitDataset(Dataset):
    def __init__(self, split_pt_path, transform=None, validate=True, validate_limit=None, validate_stats=True):
        super(PackedSplitDataset, self).__init__()
        self.split_pt_path = split_pt_path
        self.transform = transform

        if not os.path.isfile(split_pt_path):
            raise FileNotFoundError(f"Packed split file not found: {split_pt_path}")

        loaded = torch.load(split_pt_path, weights_only=False, mmap=True, map_location="cpu")
        if not isinstance(loaded, dict):
            raise TypeError(
                "Packed SECL split must be a dictionary containing "
                f"feature_config/graphs/graph_stats, got {type(loaded)} at "
                f"{split_pt_path}. Re-run preprocessing and split packing."
            )
        required = {"feature_config", "graphs", "graph_stats"}
        allowed = required | {"batch_plan"}
        if not required <= set(loaded) or not set(loaded) <= allowed:
            raise ValueError(
                f"Packed split keys must contain {sorted(required)} and only "
                f"optional key 'batch_plan', "
                f"got {sorted(loaded)}"
            )
        self.feature_config = validate_feature_config(
            loaded["feature_config"], source=split_pt_path
        )
        if not isinstance(loaded["graphs"], (list, tuple)):
            raise TypeError("packed graphs must be a list")
        if not isinstance(loaded["graph_stats"], (list, tuple)):
            raise TypeError("packed graph_stats must be a list")
        self.data_list = list(loaded["graphs"])
        self.graph_stats = list(loaded["graph_stats"])
        self.batch_plan = loaded.get("batch_plan")
        if len(self.graph_stats) != len(self.data_list):
            raise ValueError("packed graph_stats must contain one row per graph")
        names = [str(getattr(data, "circuit_name", "")) for data in self.data_list]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("packed graphs must have unique non-empty circuit_name values")

        if validate:
            validate_static_schema(
                self.data_list,
                self.feature_config,
                source=split_pt_path,
                limit=validate_limit,
            )
        for index, (data, actual) in enumerate(
            zip(self.data_list, self.graph_stats) if validate_stats else ()
        ):
            expected = estimate_graph_stats(data)
            if actual != expected:
                raise ValueError(
                    f"packed graph_stats[{index}] does not match graph "
                    f"{names[index]!r}"
                )
        if self.batch_plan is not None:
            if not isinstance(self.batch_plan, dict):
                raise TypeError("batch_plan must be a dictionary")
            required_plan = {"strategy", "batch_size", "seed", "batches"}
            if set(self.batch_plan) != required_plan:
                raise ValueError(
                    f"batch_plan keys must be exactly {sorted(required_plan)}"
                )
            batch_size = int(self.batch_plan["batch_size"])
            batches = self.batch_plan["batches"]
            if batch_size < 1 or not isinstance(batches, list):
                raise ValueError("batch_plan has invalid batch_size or batches")
            flattened = []
            for batch_id, batch in enumerate(batches):
                if not isinstance(batch, list) or not batch:
                    raise ValueError(f"batch_plan batch {batch_id} is empty or invalid")
                if len(batch) > batch_size:
                    raise ValueError(f"batch_plan batch {batch_id} exceeds batch_size")
                if batch_id + 1 < len(batches) and len(batch) != batch_size:
                    raise ValueError("only the final batch may be partial")
                flattened.extend(int(index) for index in batch)
            if sorted(flattened) != list(range(len(self.data_list))):
                raise ValueError("batch_plan must cover every graph exactly once")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        if self.transform:
            data = self.transform(data)
        values = {
            name: getattr(data, name)
            for name in _SINGLE_CHANNEL_BATCH_FIELDS
            if hasattr(data, name)
        }
        values["num_nodes"] = int(data.num_nodes)
        return AIGData(**values)
