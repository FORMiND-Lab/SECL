from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from utils.aig_schema import GateType


_PROTOTYPE_EPS = 1e-6
LOSS_TYPE = "secl_bounded_tanh_gradient_smooth_singleton_v4"


def tanh_control_parameters(m: float, delta: float) -> tuple[float, float, float]:
    limits = torch.finfo(torch.float32)
    if not math.isfinite(m) or not 1 <= m <= limits.max:
        raise ValueError("grad_scale_m must be finite, >= 1 and representable as FP32")
    if not math.isfinite(delta) or not limits.tiny <= delta <= limits.max:
        raise ValueError("grad_scale_delta must be finite and representable as positive FP32")
    log_m = math.log(m)
    lower = math.exp(log_m * math.tanh(-0.1 / delta))
    upper = math.exp(log_m * math.tanh(0.2 / delta))
    return log_m, lower, upper


def bounded_tanh_multiplier(cosine, positive, log_m, delta, lower, upper):
    dtype = torch.float64 if cosine.dtype == torch.float64 else torch.float32
    magnitude = cosine.detach().to(dtype).abs().clamp(0, 1)
    distance = 0.9 - magnitude if positive else magnitude - 0.8
    return ((distance / delta).tanh() * log_m).exp().clamp(lower, upper)


class _ScaleSimilarityGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, multiplier):
        ctx.save_for_backward(multiplier.detach())
        return value

    @staticmethod
    def backward(ctx, grad_output):
        (multiplier,) = ctx.saved_tensors
        return grad_output * multiplier, None


@dataclass(frozen=True)
class SimulationSECLResult:
    loss: torch.Tensor
    secl_loss: torch.Tensor
    singleton_negative_loss: torch.Tensor
    active_graphs: int
    singleton_negative_graphs: int
    simulation_classes: int
    singleton_classes: int
    candidate_nodes: int
    anchors: int
    entities: int
    prototypes: int
    singleton_keys: int
    relations: int
    singleton_relations: int
    singleton_pair_relations: int

    singleton_violation_relations: int
    prototype_fallbacks: int
    leave_one_out_fallbacks: int
    streamed_graphs: int
    streamed_singleton_classes: int

    @property
    def valid_anchors(self) -> int:
        return self.anchors


def prepare_secl_labels(y_index, y_sign, sim_class, batch_vec, gate, y_known=None, **unused):
    known = torch.ones_like(y_index, dtype=torch.bool) if y_known is None else y_known
    base = torch.where(known & ((gate.reshape(-1) == GateType.CONST) | (gate.reshape(-1) == GateType.AND)))[0]
    keys, inv, sizes = torch.unique(sim_class[base], return_inverse=True, return_counts=True)
    candidates = base[sizes[inv] >= 2]
    admitted = keys[sizes >= 2]
    labels = y_index[candidates]
    entities, inverse, entity_sizes = torch.unique(labels, return_inverse=True, return_counts=True)
    slots = torch.where(entity_sizes >= 2)[0]
    members = (entity_sizes >= 2)[inverse]
    indices = candidates[members]
    slot_to_proto = torch.full_like(entities, -1)
    slot_to_proto[slots] = torch.arange(len(slots), device=slots.device)
    proto_inverse = slot_to_proto[inverse[members]]
    order = torch.argsort(proto_inverse, stable=True)
    prototype_sizes = entity_sizes[slots]
    starts = prototype_sizes.cumsum(0)-prototype_sizes
    first, second = order[starts][proto_inverse], order[starts+1][proto_inverse]
    fallback = torch.where(torch.arange(len(indices), device=indices.device) == first, second, first)
    graphs = int(batch_vec[-1])+1 if len(batch_vec) else 0
    member_counts = torch.bincount(batch_vec[indices], minlength=graphs)
    entity_counts = torch.bincount(batch_vec[entities], minlength=graphs)
    prototype_counts = torch.bincount(batch_vec[entities[slots]], minlength=graphs)
    mc, ec = member_counts.tolist(), entity_counts.tolist()
    singletons = candidates[~members]
    used_singletons = singletons[member_counts[batch_vec[singletons]] > 0]
    positions = inverse[members]-torch.repeat_interleave(entity_counts.cumsum(0)-entity_counts, member_counts)
    query_tiles, key_tiles, query_offset, key_offset = [], [], 0, 0
    for n, e in zip(mc, ec):
        query_tiles.extend((start, query_offset+n, key_offset, key_offset+e)
                           for start in range(query_offset, query_offset+n, 64))
        if n:
            key_tiles.extend((start, key_offset+e, query_offset, query_offset+n, key_offset)
                             for start in range(key_offset, key_offset+e, 64))
        query_offset += n
        key_offset += e
    return dict(candidate_indices=candidates, admitted_sim_classes=admitted, entity_labels=entities,
                entity_sizes=entity_sizes, prototype_slots=slots, member_indices=indices,
                exact_singleton_indices=singletons, member_labels=labels[members], member_signs=y_sign[indices],
                proto_inverse=proto_inverse, member_order=order, prototype_sizes=prototype_sizes,
                fallback_positions=fallback, graph_count=graphs, member_counts=member_counts,
                entity_counts=entity_counts, singleton_counts=entity_counts-prototype_counts,
                range_table=[(i, n, ec[i]) for i, n in enumerate(mc) if n],
                member_count_list=mc, entity_count_list=ec, used_singleton_indices=used_singletons,
                positive_positions=positions,
                query_tiles=torch.tensor(query_tiles, dtype=torch.long, device=y_index.device).reshape(-1, 4),
                key_tiles=torch.tensor(key_tiles, dtype=torch.long, device=y_index.device).reshape(-1, 5))


def _normalized_states(value):
    return F.normalize(value, p=2, dim=1)


def _prototype_states(states, members, signs, inverse, order, sizes, representatives, fallback_positions):
    aligned = members*signs.to(states.dtype)[:, None]
    sums = torch.segment_reduce(aligned.index_select(0, order), 'sum', lengths=sizes)
    norms = sums.norm(dim=1)
    degenerate = norms <= _PROTOTYPE_EPS
    prototypes = torch.where(degenerate[:, None], states.index_select(0, representatives),
                             sums/norms.clamp_min(_PROTOTYPE_EPS)[:, None])
    remaining = sums.index_select(0, inverse)-aligned
    remaining_norm = remaining.norm(dim=1)
    fallback = remaining_norm <= _PROTOTYPE_EPS
    positives = torch.where(fallback[:, None], aligned.index_select(0, fallback_positions),
                            remaining/remaining_norm.clamp_min(_PROTOTYPE_EPS)[:, None])
    return prototypes, positives, degenerate, fallback


_compiled_normalize = torch.compile(_normalized_states, dynamic=True, fullgraph=True)
_compiled_prototypes = torch.compile(_prototype_states, dynamic=True, fullgraph=True)


class SimulationClassSECLLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 0.3,
        singleton_circle_gamma: float = 32.0,
        singleton_negative_weight: float = 1.0,
        singleton_pair_chunk_size: int = 1_048_576,
        anchor_chunk_size: int = 4096,
        key_chunk_size: int = 65536,
        checkpoint_free_bytes: int = 0,
        validate_inputs: bool = True,
        grad_scale_m: float = 2.0,
        grad_scale_delta: float = 0.1,
        singleton_similarity_center: float = 0.8,
        singleton_softplus_width: float = 0.1,
        compile_blocks: bool = False,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if singleton_circle_gamma <= 0:
            raise ValueError("singleton circle gamma must be positive")
        if singleton_negative_weight < 0:
            raise ValueError("singleton negative weight must be non-negative")
        if singleton_pair_chunk_size < 1:
            raise ValueError("singleton pair chunk size must be positive")
        if anchor_chunk_size < 1 or key_chunk_size < 1:
            raise ValueError("SECL chunk sizes must be positive")
        if checkpoint_free_bytes < 0:
            raise ValueError("checkpoint_free_bytes must be non-negative")
        log_m, lower, upper = tanh_control_parameters(grad_scale_m, grad_scale_delta)
        if not math.isfinite(singleton_similarity_center) or not 0 <= singleton_similarity_center <= 1:
            raise ValueError("singleton_similarity_center must lie in [0, 1]")
        if not math.isfinite(singleton_softplus_width) or singleton_softplus_width <= 0:
            raise ValueError("singleton_softplus_width must be finite and positive")

        self.temperature = float(temperature)
        self.margin = float(margin)
        self.singleton_circle_gamma = float(singleton_circle_gamma)
        self.singleton_negative_weight = float(singleton_negative_weight)
        self.singleton_pair_chunk_size = int(singleton_pair_chunk_size)
        self.anchor_chunk_size = int(anchor_chunk_size)
        self.key_chunk_size = int(key_chunk_size)
        self.checkpoint_free_bytes = int(checkpoint_free_bytes)
        self.validate_inputs = bool(validate_inputs)
        self.grad_scale_m = float(grad_scale_m)
        self.grad_scale_delta = float(grad_scale_delta)
        self._grad_log_m = log_m
        self.grad_scale_min = lower
        self.grad_scale_max = upper
        self.singleton_similarity_center = float(singleton_similarity_center)
        self.singleton_softplus_width = float(singleton_softplus_width)
        self.compile_blocks = bool(compile_blocks)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th_pos = -math.cos(margin)
        self.mm = math.sin(margin) * margin
        self._checkpoint_free_remaining = 0

    @staticmethod
    def _validate(
        embeddings: torch.Tensor,
        y_index: torch.Tensor,
        y_sign: torch.Tensor,
        y_known: torch.Tensor,
        sim_class: torch.Tensor,
        sim_class_sign: torch.Tensor,
        batch_vec: torch.Tensor,
        gate: torch.Tensor,
    ) -> None:
        if embeddings.dim() != 2:
            raise ValueError("embeddings must have shape [N, D]")
        node_count = embeddings.size(0)
        for name, value in {
            "y_index": y_index,
            "y_sign": y_sign,
            "sim_class": sim_class,
            "sim_class_sign": sim_class_sign,
            "batch_vec": batch_vec,
            "gate": gate.reshape(-1),
        }.items():
            if value.shape != (node_count,) or value.dtype != torch.long:
                raise ValueError(f"{name} must be node-aligned int64 [N]")
        if y_known.shape != (node_count,) or y_known.dtype != torch.bool:
            raise ValueError("y_known must be node-aligned bool [N]")
        if bool(((y_sign != -1) & (y_sign != 1)).any()):
            raise ValueError("y_sign must contain only +/-1")
        if bool(((sim_class_sign != -1) & (sim_class_sign != 1)).any()):
            raise ValueError("sim_class_sign must contain only +/-1")
        if node_count == 0:
            return
        if int(batch_vec[0]) != 0 or bool(
            (batch_vec[1:] < batch_vec[:-1]).any()
        ):
            raise ValueError("batch_vec must contain ordered graph IDs")
        for name, representatives in {
            "y_index": y_index,
            "sim_class": sim_class,
        }.items():
            if bool(((representatives < 0) | (representatives >= node_count)).any()):
                raise ValueError(f"{name} representatives must lie in [0, N)")
            unique = torch.unique(representatives)
            if not torch.equal(representatives.index_select(0, unique), unique):
                raise ValueError(f"{name} representatives must be self-labelled")
            if not torch.equal(batch_vec, batch_vec.index_select(0, representatives)):
                raise ValueError(f"{name} representatives must not cross graphs")
        if not bool((y_sign.index_select(0, torch.unique(y_index)) == 1).all()):
            raise ValueError("exact representatives must have sign +1")
        if not bool(
            (sim_class_sign.index_select(0, torch.unique(sim_class)) == 1).all()
        ):
            raise ValueError("simulation representatives must have sign +1")

    @staticmethod
    def _work_dtype(value: torch.Tensor) -> torch.dtype:
        return (
            torch.float32
            if value.dtype in (torch.float16, torch.bfloat16)
            else value.dtype
        )

    def _positive_margin(self, cosine: torch.Tensor) -> torch.Tensor:
        if self.margin == 0:
            return cosine


        sine = torch.sqrt((1.0 - cosine.square()).clamp_min(1e-6))
        angular = cosine * self.cos_m - sine * self.sin_m
        return torch.where(cosine > self.th_pos, angular, cosine - self.mm)

    def _gradient_multiplier(self, cosine: torch.Tensor, *, positive: bool) -> torch.Tensor:
        return bounded_tanh_multiplier(
            cosine, positive, self._grad_log_m, self.grad_scale_delta,
            self.grad_scale_min, self.grad_scale_max,
        )

    def _scale_similarity(self, cosine: torch.Tensor, *, positive: bool) -> torch.Tensor:
        return _ScaleSimilarityGradient.apply(
            cosine, self._gradient_multiplier(cosine, positive=positive)
        )

    def _singleton_mass(self, cosine: torch.Tensor) -> torch.Tensor:
        width = self.singleton_softplus_width
        residual = width * F.softplus((cosine.abs() - self.singleton_similarity_center) / width)
        return torch.expm1(self.singleton_circle_gamma * residual.detach() * residual)

    def _logit_block(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        positive_logits: torch.Tensor,
        positive_key_positions: torch.Tensor,
    ) -> torch.Tensor:
        cosine = (queries @ keys.t()).to(self._work_dtype(queries)).clamp(
            -1.0 + 1e-4, 1.0 - 1e-4
        )
        cosine = self._scale_similarity(cosine, positive=False)
        logits = cosine.square() / self.temperature


        columns = torch.arange(keys.size(0), device=keys.device)
        is_positive = columns[None, :] == positive_key_positions[:, None]
        logits = torch.where(is_positive, positive_logits[:, None], logits)
        return torch.logsumexp(logits, dim=1)


    _RETAINED_PLANES = 5

    def _run_block(self, *args: torch.Tensor) -> torch.Tensor:
        if self.compile_blocks and args[0].is_cuda:
            from .fused_secl import denominator
            return denominator(
                *args, self.temperature, self._grad_log_m, self.grad_scale_delta,
                self.grad_scale_min, self.grad_scale_max,
            )
        pairs = int(args[0].size(0) * args[1].size(0))
        plane = 8 if args[0].dtype == torch.float64 else 4
        retained_bytes = pairs * (self._RETAINED_PLANES * plane + 1)
        keep_activations = retained_bytes <= self._checkpoint_free_remaining
        if keep_activations:
            self._checkpoint_free_remaining -= retained_bytes
            return self._logit_block(*args)
        if torch.is_grad_enabled() and any(value.requires_grad for value in args):
            return checkpoint(self._logit_block, *args, use_reentrant=False)
        return self._logit_block(*args)

    def _singleton_pair_block(
        self,
        states: torch.Tensor,
        left_indices: torch.Tensor,
        right_indices: torch.Tensor,
        pair_graphs: torch.Tensor,
        graph_zero: torch.Tensor,
    ) -> torch.Tensor:
        left = states.index_select(0, left_indices)
        right = states.index_select(0, right_indices)
        cosine = (left * right).sum(dim=1).to(self._work_dtype(states)).clamp(
            -1.0 + 1e-4, 1.0 - 1e-4
        )
        cosine = self._scale_similarity(cosine, positive=False)

        above_reference = cosine.detach().abs() > self.singleton_similarity_center
        pair_mass = self._singleton_mass(cosine)
        mass = graph_zero.clone().index_add(
            0, pair_graphs, pair_mass
        )
        violations = graph_zero.clone().index_add(
            0, pair_graphs, above_reference.to(graph_zero.dtype)
        )
        return torch.stack((mass, violations))

    def _run_singleton_pair_block(self, *args: torch.Tensor) -> torch.Tensor:
        pairs = int(args[1].numel())
        plane = 8 if args[0].dtype == torch.float64 else 4
        dimensions = int(args[0].size(1))
        retained_bytes = pairs * ((2 * dimensions + 7) * plane + 1)
        keep_activations = retained_bytes <= self._checkpoint_free_remaining
        if keep_activations:
            self._checkpoint_free_remaining -= retained_bytes
            return self._singleton_pair_block(*args)
        if torch.is_grad_enabled() and args[0].requires_grad:
            return checkpoint(
                self._singleton_pair_block, *args, use_reentrant=False
            )
        return self._singleton_pair_block(*args)

    def _singleton_matrix_block(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        cosine = (queries @ keys.t()).to(self._work_dtype(queries)).clamp(
            -1.0 + 1e-4, 1.0 - 1e-4
        )
        cosine = self._scale_similarity(cosine, positive=False)
        upper = key_positions[None, :] > query_positions[:, None]
        pair_mass = self._singleton_mass(cosine)
        mass = (pair_mass * upper.to(pair_mass.dtype)).sum()
        above_reference = upper & (cosine.detach().abs() > self.singleton_similarity_center)
        violations = above_reference.sum().to(pair_mass.dtype)
        return torch.stack((mass, violations))

    def _run_singleton_matrix_block(self, *args: torch.Tensor) -> torch.Tensor:
        pairs = int(args[0].size(0) * args[1].size(0))
        plane = 8 if args[0].dtype == torch.float64 else 4
        retained_bytes = pairs * ((self._RETAINED_PLANES + 2) * plane + 1)
        keep_activations = retained_bytes <= self._checkpoint_free_remaining
        if keep_activations:
            self._checkpoint_free_remaining -= retained_bytes
            return self._singleton_matrix_block(*args)
        if torch.is_grad_enabled() and any(value.requires_grad for value in args):
            return checkpoint(
                self._singleton_matrix_block, *args, use_reentrant=False
            )
        return self._singleton_matrix_block(*args)

    @staticmethod
    def _materialize_segment_pairs(
        starts: torch.Tensor, counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        member_segments = torch.repeat_interleave(
            torch.arange(counts.numel(), device=counts.device), counts
        )
        local_starts = torch.cumsum(counts, dim=0) - counts
        local_rank = torch.arange(
            int(counts.sum()), device=counts.device
        ) - torch.repeat_interleave(local_starts, counts)
        positions = starts.index_select(0, member_segments) + local_rank
        repeats = counts.index_select(0, member_segments) - local_rank - 1
        left = torch.repeat_interleave(positions, repeats)
        pair_starts = torch.cumsum(repeats, dim=0) - repeats
        offsets = torch.arange(
            int(repeats.sum()), device=counts.device
        ) - torch.repeat_interleave(pair_starts, repeats)
        return left, left + 1 + offsets

    def _singleton_denominator_by_graph(
        self,
        states: torch.Tensor,
        singleton_indices: torch.Tensor,
        sim_class: torch.Tensor,
        batch_vec: torch.Tensor,
        graph_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        work_dtype = self._work_dtype(states)
        graph_mass = torch.zeros(
            graph_count, dtype=work_dtype, device=states.device
        )
        graph_pairs = torch.zeros(
            graph_count, dtype=torch.long, device=states.device
        )
        graph_violations = torch.zeros_like(graph_pairs)
        if singleton_indices.numel() < 2:
            return graph_mass, graph_pairs, graph_violations, 0, 0

        singleton_classes = sim_class.index_select(0, singleton_indices)
        order = torch.argsort(singleton_classes, stable=True)
        sorted_indices = singleton_indices.index_select(0, order)
        sorted_classes = singleton_classes.index_select(0, order)
        sorted_states = states.index_select(0, sorted_indices)
        sorted_graphs = batch_vec.index_select(0, sorted_indices)
        _, bucket_counts = torch.unique_consecutive(
            sorted_classes, return_counts=True
        )
        bucket_starts = torch.cumsum(bucket_counts, dim=0) - bucket_counts
        eligible = torch.where(bucket_counts >= 2)[0]
        if eligible.numel() == 0:
            return graph_mass, graph_pairs, graph_violations, 0, 0

        eligible_counts = bucket_counts.index_select(0, eligible)
        eligible_starts = bucket_starts.index_select(0, eligible)
        pair_counts = eligible_counts * (eligible_counts - 1) // 2
        bucket_graphs = sorted_graphs.index_select(0, eligible_starts)
        graph_pairs.scatter_add_(0, bucket_graphs, pair_counts)

        metadata = torch.stack(
            (eligible_starts, eligible_counts, pair_counts), dim=1
        ).cpu().tolist()
        small_groups: list[list[tuple[int, int]]] = []
        pending: list[tuple[int, int]] = []
        pending_pairs = 0
        large: list[tuple[int, int]] = []
        for start, count, pairs in metadata:
            if pairs > self.singleton_pair_chunk_size:
                if pending:
                    small_groups.append(pending)
                    pending = []
                    pending_pairs = 0
                large.append((start, count))
                continue
            if pending and pending_pairs + pairs > self.singleton_pair_chunk_size:
                small_groups.append(pending)
                pending = []
                pending_pairs = 0
            pending.append((start, count))
            pending_pairs += pairs
        if pending:
            small_groups.append(pending)

        graph_zero = graph_mass.new_zeros(graph_count)
        for group in small_groups:
            starts = torch.tensor(
                [value[0] for value in group],
                dtype=torch.long,
                device=states.device,
            )
            counts = torch.tensor(
                [value[1] for value in group],
                dtype=torch.long,
                device=states.device,
            )
            left, right = self._materialize_segment_pairs(starts, counts)
            pair_graphs = sorted_graphs.index_select(0, left)
            statistics = self._run_singleton_pair_block(
                sorted_states, left, right, pair_graphs, graph_zero
            )
            graph_mass = graph_mass + statistics[0]
            graph_violations.add_(statistics[1].detach().to(torch.long))

        streamed_classes = 0
        for start, count in large:
            end = start + count
            bucket_states = sorted_states[start:end]
            graph = sorted_graphs[start].reshape(1)
            bucket_mass = states.new_zeros((), dtype=work_dtype)
            bucket_violations = torch.zeros(
                (), dtype=torch.long, device=states.device
            )
            for query_start in range(0, count, self.anchor_chunk_size):
                query_end = min(query_start + self.anchor_chunk_size, count)
                query_positions = torch.arange(
                    query_start, query_end, device=states.device
                )
                for key_start in range(0, count, self.key_chunk_size):
                    key_end = min(key_start + self.key_chunk_size, count)
                    if key_end <= query_start:
                        continue
                    key_positions = torch.arange(
                        key_start, key_end, device=states.device
                    )
                    statistics = self._run_singleton_matrix_block(
                        bucket_states[query_start:query_end],
                        bucket_states[key_start:key_end],
                        query_positions,
                        key_positions,
                    )
                    bucket_mass = bucket_mass + statistics[0]
                    bucket_violations.add_(
                        statistics[1].detach().to(torch.long)
                    )
            graph_mass = graph_mass.index_add(0, graph, bucket_mass.reshape(1))
            graph_violations.index_add_(
                0, graph, bucket_violations.reshape(1)
            )
            streamed_classes += 1

        return (
            graph_mass,
            graph_pairs,
            graph_violations,
            len(metadata),
            streamed_classes,
        )

    def _positive_logits(self, anchor_states, positive_states, anchor_signs):
        work_dtype = self._work_dtype(anchor_states)
        with torch.autocast(device_type=anchor_states.device.type, enabled=False):
            cosine = (anchor_states.to(work_dtype)*positive_states.to(work_dtype)).sum(1).clamp(
                -1.0+1e-4, 1.0-1e-4)
            aligned = self._scale_similarity(anchor_signs.to(work_dtype)*cosine, positive=True)
            return self._positive_margin(aligned)/self.temperature

    def _pool_loss(
        self,
        anchor_states: torch.Tensor,
        anchor_labels: torch.Tensor,
        anchor_signs: torch.Tensor,
        positive_states: torch.Tensor,
        key_states: torch.Tensor,
        key_labels: torch.Tensor,
        singleton_denominator_mass: torch.Tensor,
        has_singleton_denominator: bool,
        positive_logits: torch.Tensor | None = None,
        positive_key_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_losses: list[torch.Tensor] = []
        combined_losses: list[torch.Tensor] = []
        work_dtype = self._work_dtype(anchor_states)
        if positive_key_positions is None:
            positive_key_positions = torch.searchsorted(key_labels, anchor_labels)
        if positive_logits is None:
            positive_logits = self._positive_logits(anchor_states, positive_states, anchor_signs)


        anchor_blocks = torch.split(anchor_states, self.anchor_chunk_size)
        logit_blocks = torch.split(positive_logits, self.anchor_chunk_size)
        position_blocks = torch.split(
            positive_key_positions, self.anchor_chunk_size
        )
        key_blocks = torch.split(key_states, self.key_chunk_size)
        log_independent = None
        if has_singleton_denominator:
            log_independent = singleton_denominator_mass.clamp_min(
                torch.finfo(singleton_denominator_mass.dtype).tiny
            ).log() + math.log(
                self.singleton_negative_weight
            )
        for anchors, logits, positions in zip(
            anchor_blocks, logit_blocks, position_blocks
        ):
            log_denominator = torch.full(
                (anchors.size(0),),
                -torch.inf,
                dtype=work_dtype,
                device=anchor_states.device,
            )
            key_start = 0
            for keys in key_blocks:
                block_logsumexp = self._run_block(
                    anchors, keys, logits, positions - key_start
                )
                log_denominator = torch.logaddexp(
                    log_denominator, block_logsumexp
                )
                key_start += keys.size(0)
            base_losses.append(log_denominator - logits)


            if log_independent is not None:
                combined_denominator = torch.logaddexp(
                    log_denominator, log_independent
                )
            else:
                combined_denominator = log_denominator
            combined_losses.append(combined_denominator - logits)
        return torch.cat(combined_losses), torch.cat(base_losses)

    @staticmethod
    def _global_graph_mean(
        local_sum: torch.Tensor, local_count: int
    ) -> tuple[torch.Tensor, int]:
        if not dist.is_initialized():
            return (local_sum/local_count if local_count else local_sum*0.0), local_count
        count = torch.tensor(
            local_count, dtype=torch.long, device=local_sum.device
        )
        global_sum = local_sum.detach().clone()
        global_count = count.clone()
        world_size = 1
        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        if int(global_count) == 0:
            return local_sum * 0.0, 0
        denominator = global_count.to(local_sum.dtype)
        differentiable = local_sum * world_size / denominator
        value = global_sum / denominator
        return differentiable + (value - differentiable.detach()), int(global_count)

    @staticmethod
    def _global_totals(local: torch.Tensor) -> list[int]:
        totals = local.detach().to(torch.long).clone()
        if dist.is_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return [int(value) for value in totals.cpu().tolist()]

    def forward(
        self,
        embeddings: torch.Tensor,
        y_index: torch.Tensor,
        y_sign: torch.Tensor,
        sim_class: torch.Tensor,
        sim_class_sign: torch.Tensor,
        batch_vec: torch.Tensor,
        gate: torch.Tensor,
        *,
        y_known: torch.Tensor | None = None,
        collect_diagnostics: bool = True,
        prepared: dict | None = None,
    ) -> SimulationSECLResult:
        if y_known is None:
            y_known = torch.ones_like(y_index, dtype=torch.bool)
        if self.validate_inputs:
            self._validate(
                embeddings,
                y_index,
                y_sign,
                y_known,
                sim_class,
                sim_class_sign,
                batch_vec,
                gate,
            )

        zero = embeddings.reshape(-1)[:0].sum().to(self._work_dtype(embeddings))
        if embeddings.size(0) == 0:
            return SimulationSECLResult(
                loss=zero,
                secl_loss=zero,
                singleton_negative_loss=zero,
                active_graphs=0,
                singleton_negative_graphs=0,
                simulation_classes=0,
                singleton_classes=0,
                candidate_nodes=0,
                anchors=0,
                entities=0,
                prototypes=0,
                singleton_keys=0,
                relations=0,
                singleton_relations=0,
                singleton_pair_relations=0,
                singleton_violation_relations=0,
                prototype_fallbacks=0,
                leave_one_out_fallbacks=0,
                streamed_graphs=0,
                streamed_singleton_classes=0,
            )

        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            normalize = _compiled_normalize if self.compile_blocks and embeddings.is_cuda else _normalized_states
            states = normalize(embeddings.to(self._work_dtype(embeddings)))
        layout = prepared if prepared is not None else prepare_secl_labels(
            y_index, y_sign, sim_class, batch_vec, gate, y_known)
        candidate_indices = layout['candidate_indices']
        admitted_sim_classes = layout['admitted_sim_classes']

        if self.validate_inputs and candidate_indices.numel():
            candidate_labels = y_index.index_select(0, candidate_indices)
            _, inverse, counts = torch.unique(
                candidate_labels, return_inverse=True, return_counts=True
            )
            proved = candidate_indices[counts.index_select(0, inverse) >= 2]
            representatives = y_index.index_select(0, proved)
            exact_relation = y_sign.index_select(0, proved) * y_sign.index_select(
                0, representatives
            )
            simulation_relation = sim_class_sign.index_select(
                0, proved
            ) * sim_class_sign.index_select(0, representatives)
            if not torch.equal(exact_relation, simulation_relation):
                raise ValueError(
                    "proved equivalence polarity conflicts with sim_class_sign"
                )
            if not torch.equal(
                sim_class.index_select(0, proved),
                sim_class.index_select(0, representatives),
            ):
                raise ValueError("a proved exact class crosses simulation classes")

        self._checkpoint_free_remaining = self.checkpoint_free_bytes
        graph_losses: list[torch.Tensor] = []
        base_graph_losses: list[torch.Tensor] = []


        entity_labels = layout['entity_labels']
        entity_sizes = layout['entity_sizes']
        prototype_slots = layout['prototype_slots']
        member_indices = layout['member_indices']
        exact_singleton_indices = layout['exact_singleton_indices']
        member_labels = layout['member_labels']
        member_signs = layout['member_signs']


        member_states = states.index_select(0, member_indices)

        if prototype_slots.numel():
            build_prototypes = _compiled_prototypes if self.compile_blocks and states.is_cuda else _prototype_states
            prototypes, positive_states, degenerate, leave_one_out_degenerate = build_prototypes(
                states, member_states, member_signs, layout['proto_inverse'], layout['member_order'],
                layout['prototype_sizes'], entity_labels.index_select(0, prototype_slots), layout['fallback_positions'])
        else:
            degenerate = torch.empty(
                0, dtype=torch.bool, device=embeddings.device
            )
            leave_one_out_degenerate = torch.empty(
                0, dtype=torch.bool, device=embeddings.device
            )
            prototypes = states.new_empty((0, states.size(1)))
            positive_states = states.new_empty((0, states.size(1)))

        entity_states = states.index_select(0, entity_labels)
        if prototype_slots.numel():
            entity_states = entity_states.index_copy(
                0, prototype_slots, prototypes
            )

        graph_count = layout['graph_count']
        member_counts = layout['member_counts']
        entity_counts = layout['entity_counts']
        singleton_counts = layout['singleton_counts']
        range_table = layout['range_table']

        if self.singleton_negative_weight > 0:
            used_singleton_indices = layout['used_singleton_indices']
            (
                singleton_denominator_mass,
                singleton_pair_counts,
                singleton_violation_counts,
                singleton_class_count,
                streamed_singleton_classes,
            ) = self._singleton_denominator_by_graph(
                states,
                used_singleton_indices,
                sim_class,
                batch_vec,
                graph_count,
            )
        else:
            singleton_denominator_mass = states.new_zeros(
                graph_count, dtype=self._work_dtype(states)
            )
            singleton_pair_counts = torch.zeros(
                graph_count, dtype=torch.long, device=embeddings.device
            )
            singleton_violation_counts = torch.zeros_like(
                singleton_pair_counts
            )
            singleton_class_count = 0
            streamed_singleton_classes = 0
        singleton_graph_mask = (
            (singleton_pair_counts > 0) & (member_counts > 0)
        )
        active_graph_mask = member_counts > 0
        singleton_pair_count_list = singleton_pair_counts.tolist()

        relation_total = torch.zeros(
            (), dtype=torch.long, device=embeddings.device
        )
        singleton_relation_total = torch.zeros_like(relation_total)
        if (self.compile_blocks and states.is_cuda and states.dtype == torch.float32
            and torch.is_autocast_enabled('cuda') and torch.get_autocast_dtype('cuda') == torch.bfloat16
            and member_indices.numel()):
            from .fused_secl import circuit_denominator
            positive = self._positive_logits(member_states, positive_states, member_signs)
            normalizer = circuit_denominator(member_states, entity_states, positive, layout['positive_positions'],
                layout['query_tiles'], layout['key_tiles'], self.temperature, self._grad_log_m,
                self.grad_scale_delta, self.grad_scale_min, self.grad_scale_max)
            base_values = normalizer-positive
            if self.singleton_negative_weight > 0:
                member_graphs = batch_vec.index_select(0, member_indices)
                log_mass = singleton_denominator_mass.clamp_min(torch.finfo(states.dtype).tiny).log()
                log_mass = log_mass+math.log(self.singleton_negative_weight)
                combined = torch.logaddexp(normalizer, log_mass.index_select(0, member_graphs))
                values = torch.where(singleton_graph_mask.index_select(0, member_graphs), combined, normalizer)-positive
            else:
                values = base_values
            count = member_counts.clamp_min(1)
            graph_losses = list(torch.unbind((torch.segment_reduce(values, 'sum', lengths=member_counts)/count)[active_graph_mask]))
            base_graph_losses = list(torch.unbind((torch.segment_reduce(base_values, 'sum', lengths=member_counts)/count)[active_graph_mask]))
            if collect_diagnostics:
                relation_total = (member_counts*(entity_counts-1)).sum()
                singleton_relation_total = (member_counts*singleton_counts).sum()
        else:
            member_count_list = layout['member_count_list']
            entity_count_list = layout['entity_count_list']
            anchor_state_chunks = torch.split(member_states, member_count_list)
            anchor_label_chunks = torch.split(member_labels, member_count_list)
            anchor_sign_chunks = torch.split(member_signs, member_count_list)
            positive_state_chunks = torch.split(positive_states, member_count_list)
            positive_logit_chunks = torch.split(
                self._positive_logits(member_states, positive_states, member_signs), member_count_list
            )
            key_state_chunks = torch.split(entity_states, entity_count_list)
            key_label_chunks = torch.split(entity_labels, entity_count_list)
            position_chunks = torch.split(layout['positive_positions'], member_count_list)
            for graph, anchor_count, key_count in range_table:
                per_anchor, base_per_anchor = self._pool_loss(
                    anchor_state_chunks[graph],
                    anchor_label_chunks[graph],
                    anchor_sign_chunks[graph],
                    positive_state_chunks[graph],
                    key_state_chunks[graph],
                    key_label_chunks[graph],
                    singleton_denominator_mass[graph],
                    singleton_pair_count_list[graph] > 0,
                    positive_logits=positive_logit_chunks[graph],
                    positive_key_positions=position_chunks[graph],
                )
                graph_losses.append(per_anchor.mean())
                base_graph_losses.append(base_per_anchor.mean())
                if collect_diagnostics:
                    relation_total += anchor_count * (key_count - 1)
                    singleton_relation_total += (
                        anchor_count * singleton_counts[graph]
                    )

        counters = torch.stack(
            (
                torch.as_tensor(
                    candidate_indices.numel(), device=embeddings.device
                ),
                torch.as_tensor(member_indices.numel(), device=embeddings.device),
                entity_counts.index_select(
                    0, torch.where(active_graph_mask)[0]
                ).sum(),
                torch.as_tensor(
                    prototype_slots.numel(), device=embeddings.device
                ),
                singleton_counts.index_select(
                    0, torch.where(active_graph_mask)[0]
                ).sum(),
                relation_total,
                singleton_relation_total,
                singleton_pair_counts.sum(),
                singleton_violation_counts.sum(),
                degenerate.sum(),
                leave_one_out_degenerate.sum(),
                torch.as_tensor(
                    len(range_table), device=embeddings.device
                ),
                torch.as_tensor(
                    singleton_class_count, device=embeddings.device
                ),
                torch.as_tensor(
                    streamed_singleton_classes, device=embeddings.device
                ),
            )
        ).to(torch.long) if collect_diagnostics else None

        combined_local_sum = (
            torch.stack(graph_losses).sum() if graph_losses else zero
        )
        base_local_sum = (
            torch.stack(base_graph_losses).sum()
            if base_graph_losses
            else zero
        )
        loss, active_graphs = self._global_graph_mean(
            combined_local_sum, len(graph_losses)
        )
        secl_loss, _ = self._global_graph_mean(
            base_local_sum, len(base_graph_losses)
        )
        singleton_negative_loss = loss - secl_loss
        singleton_negative_graphs = self._global_totals(
            singleton_graph_mask.sum().reshape(1)
        )[0]
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                "prototype SECL produced a non-finite loss; do not continue "
                "or checkpoint this optimizer state"
            )

        if collect_diagnostics:
            totals = self._global_totals(counters)
            simulation_classes = self._global_totals(
                torch.tensor(
                    [admitted_sim_classes.numel()],
                    dtype=torch.long,
                    device=embeddings.device,
                )
            )[0]
        else:
            totals = [0] * 14
            simulation_classes = 0
        return SimulationSECLResult(
            loss=loss,
            secl_loss=secl_loss,
            singleton_negative_loss=singleton_negative_loss,
            active_graphs=active_graphs,
            singleton_negative_graphs=singleton_negative_graphs,
            simulation_classes=simulation_classes,
            singleton_classes=totals[12],
            candidate_nodes=totals[0],
            anchors=totals[1],
            entities=totals[2],
            prototypes=totals[3],
            singleton_keys=totals[4],
            relations=totals[5],
            singleton_relations=totals[6],
            singleton_pair_relations=totals[7],
            singleton_violation_relations=totals[8],
            prototype_fallbacks=totals[9],
            leave_one_out_fallbacks=totals[10],
            streamed_graphs=totals[11],
            streamed_singleton_classes=totals[13],
        )


LOSS_DEFAULTS = {
    "grad_scale_m": 2.0,
    "grad_scale_delta": 0.1,
    "singleton_similarity_center": 0.8,
    "singleton_softplus_width": 0.1,
}


def loss_hyperparameters(args):
    values = dict(vars(args))
    if values.get("loss_version", "current") != "current":
        raise ValueError("This release trains the current SECL objective only")
    if values.get("loss_type", LOSS_TYPE) != LOSS_TYPE:
        raise ValueError("loss_type does not match the SECL objective")
    values.update(loss_version="current", loss_type=LOSS_TYPE)
    for key, default in LOSS_DEFAULTS.items():
        values.setdefault(key, default)
    return values


def validate_loss_checkpoint(saved, current):
    if saved.get("loss_type") != LOSS_TYPE:
        raise RuntimeError(
            "Checkpoint loss_type/gradient rule differs from SECL; "
            "use --checkpoint to initialize encoder weights for a new run"
        )

    if saved.get("loss_version") not in (None, "current", "secl_tanh_v4"):
        raise RuntimeError("Checkpoint loss_version differs from SECL")
    for key in LOSS_DEFAULTS:
        if saved.get(key) != current[key]:
            raise RuntimeError(
                f"Checkpoint {key} is incompatible: expected {current[key]!r}, "
                f"found {saved.get(key)!r}"
            )


def build_training_loss(args):
    params = loss_hyperparameters(args)
    return SimulationClassSECLLoss(
        temperature=getattr(args, "stage2_temperature", 0.1),
        margin=getattr(args, "stage2_margin", 0.3),
        singleton_circle_gamma=getattr(args, "singleton_circle_gamma", 32.0),
        singleton_negative_weight=getattr(args, "singleton_negative_weight", 1.0),
        singleton_pair_chunk_size=getattr(args, "singleton_pair_chunk_size", 1_048_576),
        anchor_chunk_size=args.anchor_chunk_size,
        key_chunk_size=args.key_chunk_size,
        checkpoint_free_bytes=getattr(args, "checkpoint_free_bytes", 0),
        validate_inputs=False,
        compile_blocks=getattr(args, "secl_compile", False),
        **{key: params[key] for key in LOSS_DEFAULTS},
    )
