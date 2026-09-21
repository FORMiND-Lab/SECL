from __future__ import annotations

import hashlib

from collections.abc import Mapping, Sequence

from typing import Any, Dict, Iterable, Optional

import torch

import numpy as np

try:
    from .aig_schema import GateType
except ImportError:
    from aig_schema import GateType

TOKEN_BYTES = 16

PE_DIM = TOKEN_BYTES * 8


_DOMAIN = b"GateNexus.StructuralPE"

DEPLOYMENT_FEATURES = ("functional_signature",)

EVALUATION_FEATURES = ("sim_class", "sim_class_sign")

ALL_STATIC_FEATURES = DEPLOYMENT_FEATURES + EVALUATION_FEATURES

NODE_ID_FEATURES = ("sim_class",)

EDGE_MAJOR_FEATURES: tuple[str, ...] = ()

def build_feature_config() -> dict[str, Any]:
    return {
        "token_bytes": TOKEN_BYTES,
        "simulation_bits": PE_DIM,
        "feature_names": list(ALL_STATIC_FEATURES),
    }

def _frame(tag: bytes, parts: Iterable[bytes]) -> bytes:
    payload = bytearray(_DOMAIN)
    payload.extend(len(tag).to_bytes(2, "big"))
    payload.extend(tag)
    for part in parts:
        payload.extend(len(part).to_bytes(4, "big"))
        payload.extend(part)
    return bytes(payload)

def _encode_pi_identity(identity: Any) -> bytes:
    if isinstance(identity, torch.Tensor):
        if identity.numel() != 1:
            raise ValueError("Each PI identity tensor must contain exactly one value")
        identity = identity.item()
    if isinstance(identity, bool):
        return b"bool:" + (b"1" if identity else b"0")
    if isinstance(identity, int):
        return b"int:" + str(identity).encode("ascii")
    if isinstance(identity, str):
        return b"str:" + identity.encode("utf-8")
    if isinstance(identity, bytes):
        return b"bytes:" + identity
    raise TypeError(
        "PI identities must be int, str, bytes, bool, or scalar tensors; "
        f"got {type(identity).__name__}"
    )

def _pi_identity_map(pi_nodes: list[int], pi_ids: Optional[Any], num_nodes: int) -> Dict[int, Any]:
    if pi_ids is None:
        return {node: ordinal for ordinal, node in enumerate(pi_nodes)}
    if isinstance(pi_ids, Mapping):
        missing = [node for node in pi_nodes if node not in pi_ids]
        if missing:
            raise ValueError(f"pi_ids mapping is missing PI node(s): {missing[:5]}")
        return {node: pi_ids[node] for node in pi_nodes}

    if isinstance(pi_ids, torch.Tensor):
        values = pi_ids.detach().cpu().reshape(-1).tolist()
    elif isinstance(pi_ids, Sequence) and not isinstance(pi_ids, (str, bytes)):
        values = list(pi_ids)
    else:
        raise TypeError("pi_ids must be a mapping, sequence, tensor, or None")

    if len(values) == len(pi_nodes):
        return {node: values[ordinal] for ordinal, node in enumerate(pi_nodes)}
    if len(values) == num_nodes:
        return {node: values[node] for node in pi_nodes}
    raise ValueError(
        f"pi_ids has length {len(values)}; expected {len(pi_nodes)} (PI-only) "
        f"or {num_nodes} (node-aligned)"
    )

def _register_digest(registry: Dict[bytes, bytes], descriptor: bytes) -> bytes:
    digest = hashlib.blake2b(descriptor, digest_size=TOKEN_BYTES).digest()
    previous = registry.get(digest)
    if previous is not None and previous != descriptor:
        raise RuntimeError(
            "Detected a BLAKE2b-128 structural-token collision. "
            "Refusing to alias two different canonical structures."
        )
    registry[digest] = descriptor
    return digest

def build_structural_tokens(
    gate: torch.Tensor,
    edge_index: torch.Tensor,
    forward_level: torch.Tensor,
    forward_index: torch.Tensor,
    pi_ids: Optional[Any] = None,
) -> torch.Tensor:
    if not isinstance(gate, torch.Tensor):
        gate = torch.as_tensor(gate)
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index)
    if not isinstance(forward_level, torch.Tensor):
        forward_level = torch.as_tensor(forward_level)
    if not isinstance(forward_index, torch.Tensor):
        forward_index = torch.as_tensor(forward_index)

    output_device = gate.device
    gates = gate.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    levels = forward_level.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    order_hint = forward_index.detach().reshape(-1).to(device="cpu", dtype=torch.long)
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)

    num_nodes = int(gates.numel())
    if levels.numel() != num_nodes:
        raise ValueError(f"forward_level has {levels.numel()} entries for {num_nodes} nodes")
    if order_hint.numel() != num_nodes or sorted(order_hint.tolist()) != list(range(num_nodes)):
        raise ValueError("forward_index must be a permutation of all node indices")
    if edges.dim() != 2 or edges.size(0) != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edges.shape)}")
    if num_nodes == 0:
        return torch.empty((0, TOKEN_BYTES), dtype=torch.uint8, device=output_device)
    if torch.any(levels < 0):
        bad = torch.where(levels < 0)[0].tolist()
        raise ValueError(f"Graph is not a valid DAG; unassigned forward levels at nodes {bad[:5]}")

    fanins: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in edges.t().tolist():
        if src < 0 or src >= num_nodes or dst < 0 or dst >= num_nodes:
            raise ValueError(f"edge_index contains out-of-range edge ({src}, {dst})")
        fanins[dst].append(src)

    pi_nodes = sorted(node for node in range(num_nodes) if int(gates[node]) == 0)
    identities = _pi_identity_map(pi_nodes, pi_ids, num_nodes)
    encoded_identities: Dict[int, bytes] = {}
    identity_owners: Dict[bytes, int] = {}
    for node in pi_nodes:
        encoded = _encode_pi_identity(identities[node])
        previous_node = identity_owners.get(encoded)
        if previous_node is not None:
            raise ValueError(
                f"PI nodes {previous_node} and {node} share the same identity; "
                "PI identities must be unique within a graph"
            )
        identity_owners[encoded] = node
        encoded_identities[node] = encoded
    rank = {node: pos for pos, node in enumerate(order_hint.tolist())}
    topo_order = sorted(range(num_nodes), key=lambda node: (int(levels[node]), rank[node]))

    registry: Dict[bytes, bytes] = {}
    digests: list[Optional[bytes]] = [None] * num_nodes
    for node in topo_order:
        gate_type = int(gates[node])
        node_fanins = fanins[node]
        if gate_type == GateType.CONST:
            if node_fanins:
                raise ValueError(
                    f"CONST0 node {node} must have zero fanins, got {len(node_fanins)}"
                )
            descriptor = _frame(b"CONST0", [])
        elif gate_type == GateType.PI:
            if node_fanins:
                raise ValueError(f"PI node {node} must have zero fanins, got {len(node_fanins)}")
            descriptor = _frame(b"PI", [encoded_identities[node]])
        elif gate_type == GateType.NOT:
            if len(node_fanins) != 1:
                raise ValueError(f"NOT node {node} must have one fanin, got {len(node_fanins)}")
            child = node_fanins[0]
            if int(levels[child]) >= int(levels[node]) or digests[child] is None:
                raise ValueError(f"NOT node {node} does not follow its fanin {child} topologically")
            descriptor = _frame(b"NOT", [digests[child]])
        elif gate_type == GateType.AND:
            if len(node_fanins) != 2:
                raise ValueError(f"AND node {node} must have two fanins, got {len(node_fanins)}")
            child_digests = []
            for child in node_fanins:
                if int(levels[child]) >= int(levels[node]) or digests[child] is None:
                    raise ValueError(f"AND node {node} does not follow its fanin {child} topologically")
                child_digests.append(digests[child])
            descriptor = _frame(b"AND", sorted(child_digests))
        else:
            raise ValueError(
                f"Unsupported gate type {gate_type} at node {node}; "
                "expected PI=0, AND=1, NOT=2, CONST=3"
            )
        digests[node] = _register_digest(registry, descriptor)

    token_bytes = torch.tensor([list(digest) for digest in digests], dtype=torch.uint8)
    return token_bytes.to(output_device)

def _prepare_packed_aig(
    gate: torch.Tensor,
    edge_index: torch.Tensor,
    forward_level: torch.Tensor,
    forward_index: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[tuple[torch.Tensor, torch.Tensor], ...],
]:
    gates = torch.as_tensor(gate).detach().reshape(-1).to(
        device="cpu", dtype=torch.long
    )
    edges = torch.as_tensor(edge_index).detach().to(
        device="cpu", dtype=torch.long
    )
    levels = torch.as_tensor(forward_level).detach().reshape(-1).to(
        device="cpu", dtype=torch.long
    )
    order = torch.as_tensor(forward_index).detach().reshape(-1).to(
        device="cpu", dtype=torch.long
    )
    node_count = int(gates.numel())
    if node_count == 0:
        raise ValueError("packed AIG simulation requires at least one node")
    if levels.shape != (node_count,) or bool((levels < 0).any()):
        raise ValueError("forward_level must be non-negative int64 [N]")
    if order.shape != (node_count,) or not torch.equal(
        torch.sort(order).values, torch.arange(node_count)
    ):
        raise ValueError("forward_index must be a permutation of all node IDs")
    if edges.dim() != 2 or edges.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, E], got {tuple(edges.shape)}"
        )
    if edges.numel() and (
        bool((edges < 0).any()) or int(edges.max()) >= node_count
    ):
        raise ValueError("edge_index contains an out-of-range node ID")

    supported = (
        (gates == GateType.CONST)
        | (gates == GateType.PI)
        | (gates == GateType.NOT)
        | (gates == GateType.AND)
    )
    if not bool(supported.all()):
        raise ValueError("packed AIG simulation received an unsupported gate")

    fanin_count = torch.bincount(edges[1], minlength=node_count)
    expected_count = torch.zeros(node_count, dtype=torch.long)
    expected_count[gates == GateType.NOT] = 1
    expected_count[gates == GateType.AND] = 2
    if not torch.equal(fanin_count, expected_count):
        raise ValueError("AIG fanin counts do not match gate semantics")
    if edges.numel() and not bool(
        (
            levels.index_select(0, edges[0])
            < levels.index_select(0, edges[1])
        ).all()
    ):
        raise ValueError("every simulated AIG edge must advance the DAG level")

    fanins = torch.full((node_count, 2), -1, dtype=torch.long)
    if edges.numel():
        edge_order = torch.argsort(edges[1], stable=True)
        sorted_src = edges[0].index_select(0, edge_order)
        sorted_dst = edges[1].index_select(0, edge_order)
        starts = torch.cumsum(fanin_count, 0) - fanin_count
        one_fanin = torch.where(fanin_count == 1)[0]
        two_fanin = torch.where(fanin_count == 2)[0]
        fanins[one_fanin, 0] = sorted_src.index_select(
            0, starts.index_select(0, one_fanin)
        )
        two_starts = starts.index_select(0, two_fanin)
        fanins[two_fanin, 0] = sorted_src.index_select(0, two_starts)
        fanins[two_fanin, 1] = sorted_src.index_select(0, two_starts + 1)
        if not torch.equal(
            sorted_dst,
            torch.repeat_interleave(
                torch.arange(node_count), fanin_count
            ),
        ):
            raise RuntimeError("failed to compile packed AIG fanins")

    level_groups = []
    for level in torch.unique(levels, sorted=True).tolist():
        level_nodes = torch.where(levels == int(level))[0]
        level_groups.append(
            (
                level_nodes[gates.index_select(0, level_nodes) == GateType.AND],
                level_nodes[gates.index_select(0, level_nodes) == GateType.NOT],
            )
        )
    return gates, fanins, torch.where(gates == GateType.PI)[0], tuple(level_groups)

def _simulate_packed_aig(
    compiled: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ],
    pi_patterns: torch.Tensor,
) -> torch.Tensor:
    gates, fanins, pi_nodes, level_groups = compiled
    patterns = torch.as_tensor(pi_patterns).detach().to(
        device="cpu", dtype=torch.uint8
    )
    if patterns.dim() != 2 or patterns.size(0) != pi_nodes.numel():
        raise ValueError(
            "pi_patterns must have shape [num_pi, positive_bytes]"
        )
    if patterns.size(1) < 1:
        raise ValueError("packed simulation requires at least one pattern byte")

    state = torch.zeros(
        (gates.numel(), patterns.size(1)), dtype=torch.uint8
    )
    if pi_nodes.numel():
        state[pi_nodes] = patterns
    for and_nodes, not_nodes in level_groups:
        if and_nodes.numel():
            state[and_nodes] = torch.bitwise_and(
                state[fanins[and_nodes, 0]],
                state[fanins[and_nodes, 1]],
            )
        if not_nodes.numel():
            state[not_nodes] = torch.bitwise_not(
                state[fanins[not_nodes, 0]]
            )
    return state

def build_functional_signatures(
    gate: torch.Tensor,
    edge_index: torch.Tensor,
    forward_level: torch.Tensor,
    forward_index: torch.Tensor,
    structural_token: torch.Tensor,
) -> torch.Tensor:
    output_device = torch.as_tensor(gate).device
    compiled = _prepare_packed_aig(
        gate, edge_index, forward_level, forward_index
    )
    token = torch.as_tensor(structural_token).detach().to(
        device="cpu", dtype=torch.uint8
    )
    if token.shape != (compiled[0].numel(), TOKEN_BYTES):
        raise ValueError(
            "structural_token must have shape "
            f"[{compiled[0].numel()}, {TOKEN_BYTES}]"
        )
    signature = _simulate_packed_aig(
        compiled, token.index_select(0, compiled[2])
    )
    return signature.to(output_device)

def canonical_signature_rows(signature: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(signature).detach().to(device="cpu", dtype=torch.uint8)
    if value.dim() != 2:
        raise ValueError("signature must be a two-dimensional byte tensor")
    if value.size(1) == 0:
        raise ValueError("signature rows must contain at least one byte")


    invert = value[:, 0] >= 128
    canonical = torch.where(
        invert.unsqueeze(1), torch.bitwise_not(value), value
    )
    return canonical.to(signature.device)

def build_simulation_classes(
    signature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.as_tensor(signature).detach().to(device="cpu", dtype=torch.uint8)
    canonical = canonical_signature_rows(value).contiguous().numpy()
    row_dtype = np.dtype((np.void, canonical.shape[1]))
    packed = canonical.view(row_dtype).reshape(-1)
    _, first, inverse = np.unique(
        packed, return_index=True, return_inverse=True
    )
    representatives = first[inverse].astype(np.int64, copy=False)

    canonical_sign = np.where(
        value[:, 0].numpy() < 128,
        np.int64(1),
        np.int64(-1),
    )
    representative_sign = canonical_sign[representatives]
    relative_sign = canonical_sign * representative_sign
    device = torch.as_tensor(signature).device
    return (
        torch.from_numpy(representatives).to(device=device, dtype=torch.long),
        torch.from_numpy(relative_sign).to(device=device, dtype=torch.long),
    )

def build_single_channel_features(
    gate: torch.Tensor,
    edge_index: torch.Tensor,
    forward_level: torch.Tensor,
    forward_index: torch.Tensor,
    *,
    pi_ids: Optional[Any] = None,
    functional_signature: Optional[torch.Tensor] = None,
    sim_class: Optional[torch.Tensor] = None,
    sim_class_sign: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    gate_tensor = torch.as_tensor(gate)
    node_count = gate_tensor.numel()
    if functional_signature is None:
        structural_token = build_structural_tokens(
            gate,
            edge_index,
            forward_level,
            forward_index,
            pi_ids=pi_ids,
        )
        functional_signature = build_functional_signatures(
            gate,
            edge_index,
            forward_level,
            forward_index,
            structural_token,
        )
    else:
        functional_signature = torch.as_tensor(functional_signature)
        if pi_ids is not None:
            raise ValueError(
                "pi_ids cannot be combined with a precomputed functional_signature"
            )
    if (sim_class is None) != (sim_class_sign is None):
        raise ValueError(
            "sim_class and sim_class_sign must be provided together"
        )
    if sim_class is None:
        sim_class, sim_class_sign = build_simulation_classes(
            functional_signature
        )
    else:
        sim_class = torch.as_tensor(sim_class)
        sim_class_sign = torch.as_tensor(sim_class_sign)
    if (
        functional_signature.shape != (node_count, TOKEN_BYTES)
        or functional_signature.dtype != torch.uint8
        or sim_class.shape != (node_count,)
        or sim_class.dtype != torch.long
        or sim_class_sign.shape != (node_count,)
        or sim_class_sign.dtype != torch.long
        or bool(((sim_class_sign != 1) & (sim_class_sign != -1)).any())
    ):
        raise ValueError(
            "precomputed signature and simulation classes have invalid "
            "shape, dtype, or polarity"
        )
    features = {
        "functional_signature": functional_signature.to(gate_tensor.device),
        "sim_class": sim_class.to(gate_tensor.device),
        "sim_class_sign": sim_class_sign.to(gate_tensor.device),
    }
    if set(features) != set(ALL_STATIC_FEATURES):
        raise RuntimeError("packed feature registry is inconsistent")
    return features
