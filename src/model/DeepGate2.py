import torch
from torch import nn
from torch.nn import GRU
from .tfmlp import TFMlpAggr
from .mlp import MLP
from .dag_state import DAGState
from utils.aig_schema import GateType


def _pi_groups(forward_level, forward_index, batch, gate):
    mask = forward_level == 0
    if gate is not None:
        mask = mask & (gate.reshape(-1) == int(GateType.PI))
    positions = torch.where(mask)[0]
    nodes = forward_index.index_select(0, positions)
    if batch is None or not nodes.numel():
        return (nodes,)
    graphs = batch.index_select(0, positions)
    order = torch.argsort(graphs, stable=True)
    counts = torch.unique_consecutive(graphs.index_select(0, order), return_counts=True)[1]
    return nodes.index_select(0, order).split(counts.tolist())

def generate_orthogonal_vectors(n, dim, device='cpu'):

    if n <= dim:
        random_matrix = torch.randn(dim, dim, device=device)
        q, r = torch.linalg.qr(random_matrix)
        vectors = q[:, :n].t()
    else:
        vectors = torch.randn(n, dim, device=device)

        norms = torch.norm(vectors, dim=1, keepdim=True)
        vectors = vectors / (norms + 1e-8)

    return vectors

def generate_hs_init(
    forward_level, forward_index, hs, no_dim, batch=None, gate=None,
    pi_groups=None,
):
    device = hs.device

    if pi_groups is None:
        pi_groups = _pi_groups(forward_level, forward_index, batch, gate)
    for pi_node in pi_groups:
        if len(pi_node) > 0:
            pi_vec = generate_orthogonal_vectors(len(pi_node), no_dim, device=device)
            hs[pi_node] = pi_vec.to(dtype=hs.dtype)

    return hs

def generate_hf_init(
    forward_level,
    forward_index,
    hf,
    no_dim,
    batch=None,
    pi_init_mode='orth',
    gate=None,
    pi_groups=None,
):
    device = hf.device

    if pi_groups is None:
        pi_groups = _pi_groups(forward_level, forward_index, batch, gate)
    for pi_node in pi_groups:
        if len(pi_node) > 0:
            if pi_init_mode == 'const':
                hf[pi_node] = 0.0
            else:
                pi_vec = generate_orthogonal_vectors(len(pi_node), no_dim, device=device)
                hf[pi_node] = pi_vec.to(dtype=hf.dtype)

    return hf

def subgraph(target_idx, edge_index, edge_attr=None, dim=1):
    edge_mask = torch.isin(edge_index[dim], target_idx)


    lp_edge_index = edge_index[:, edge_mask]

    if edge_attr is not None:
        lp_edge_attr = edge_attr[edge_mask, :]
    else:
        lp_edge_attr = None

    return lp_edge_index, lp_edge_attr

class DeepGate2(nn.Module):
    def __init__(
        self,
        num_rounds=1,
        dim_hidden=128,
        enable_encode=True,
        enable_reverse=False,
        pi_init_mode='const',
    ):
        super(DeepGate2, self).__init__()


        if num_rounds < 1 or dim_hidden < 1:
            raise ValueError("num_rounds and dim_hidden must be positive")
        if pi_init_mode not in ("const", "orth"):
            raise ValueError("pi_init_mode must be 'const' or 'orth'")
        self.num_rounds = num_rounds
        self.enable_encode = enable_encode
        self.enable_reverse = enable_reverse
        self.pi_init_mode = pi_init_mode


        self.dim_hidden = dim_hidden
        self.dim_mlp = 32


        self.aggr_and_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
        self.aggr_not_strc = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)
        self.aggr_and_func = TFMlpAggr(self.dim_hidden*2, self.dim_hidden)
        self.aggr_not_func = TFMlpAggr(self.dim_hidden*1, self.dim_hidden)

        self.update_and_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_and_func = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_strc = GRU(self.dim_hidden, self.dim_hidden)
        self.update_not_func = GRU(self.dim_hidden, self.dim_hidden)


        self.readout_prob = MLP(self.dim_hidden, self.dim_mlp, 1, num_layer=3, p_drop=0.2, norm_layer='batchnorm', act_layer='relu')


        self.register_buffer(
            "_const_function_anchor",
            -torch.ones(self.dim_hidden),
            persistent=False,
        )

    @property
    def model_config(self):
        return {
            "type": "DeepGate2",
            "num_rounds": int(self.num_rounds),
            "dim_hidden": int(self.dim_hidden),
            "enable_encode": bool(self.enable_encode),
            "enable_reverse": bool(self.enable_reverse),
            "pi_init_mode": self.pi_init_mode,
        }

    def checkpoint_payload(self):
        return {
            "model_config": self.model_config,
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path):
        payload = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        if not isinstance(payload, dict) or set(payload) != {
            "model_config",
            "state_dict",
        }:
            raise RuntimeError(
                "DeepGate2 checkpoint must contain exactly model_config "
                "and state_dict"
            )
        config = payload["model_config"]
        expected_keys = {
            "type",
            "num_rounds",
            "dim_hidden",
            "enable_encode",
            "enable_reverse",
            "pi_init_mode",
        }
        if not isinstance(config, dict) or set(config) != expected_keys:
            raise RuntimeError("DeepGate2 model_config is incompatible")
        if config["type"] != "DeepGate2":
            raise RuntimeError("checkpoint does not contain DeepGate2")
        model = cls(
            num_rounds=config["num_rounds"],
            dim_hidden=config["dim_hidden"],
            enable_encode=config["enable_encode"],
            enable_reverse=config["enable_reverse"],
            pi_init_mode=config["pi_init_mode"],
        )
        if model.model_config != config:
            raise RuntimeError("DeepGate2 model_config is incompatible")
        model.load_state_dict(payload["state_dict"], strict=True)
        return model

    def pred_prob(self, hf):
        prob = self.readout_prob(hf)
        prob = torch.clamp(prob, min=0.0, max=1.0)
        return prob

    @staticmethod
    def _compact_fanin_schedule(gate, edge_index, forward_level):
        if edge_index.numel() == 0:
            return [], 0
        edge_dst = edge_index[1]
        run_start_mask = torch.ones_like(edge_dst, dtype=torch.bool)
        run_start_mask[1:] = edge_dst[1:] != edge_dst[:-1]
        run_start = torch.where(run_start_mask)[0]
        active_nodes = edge_dst.index_select(0, run_start)
        expected_active = (
            (gate == int(GateType.AND))
            | (gate == int(GateType.NOT))
        ).sum()
        if active_nodes.numel() != int(expected_active):
            permutation = torch.argsort(edge_dst, stable=True)
            edge_index = edge_index.index_select(1, permutation)
            edge_dst = edge_index[1]
            run_start_mask = torch.ones_like(edge_dst, dtype=torch.bool)
            run_start_mask[1:] = edge_dst[1:] != edge_dst[:-1]
            run_start = torch.where(run_start_mask)[0]
            active_nodes = edge_dst.index_select(0, run_start)

        active_gate = gate.index_select(0, active_nodes)
        active_level = forward_level.index_select(0, active_nodes)
        and_active = active_gate == int(GateType.AND)
        not_active = active_gate == int(GateType.NOT)
        and_nodes = active_nodes[and_active]
        and_start = run_start[and_active]
        and_level = active_level[and_active]
        not_nodes = active_nodes[not_active]
        not_start = run_start[not_active]
        not_level = active_level[not_active]
        edge_src = edge_index[0]
        num_levels = int(forward_level.max().item()) + 1
        schedule = []
        for level in range(1, num_levels):
            level_and = and_level == level
            level_and_nodes = and_nodes[level_and]
            level_and_start = and_start[level_and]
            if level_and_nodes.numel():
                and_order = torch.argsort(level_and_nodes)
                level_and_nodes = level_and_nodes.index_select(
                    0, and_order
                )
                level_and_start = level_and_start.index_select(
                    0, and_order
                )
            and_positions = torch.stack(
                (level_and_start, level_and_start + 1), dim=1
            ).reshape(-1)
            level_not = not_level == level
            level_not_nodes = not_nodes[level_not]
            level_not_start = not_start[level_not]
            if level_not_nodes.numel():
                not_order = torch.argsort(level_not_nodes)
                level_not_nodes = level_not_nodes.index_select(
                    0, not_order
                )
                level_not_start = level_not_start.index_select(
                    0, not_order
                )
            and_local_dst = torch.arange(
                level_and_nodes.numel(),
                device=gate.device,
                dtype=torch.long,
            ).repeat_interleave(2)
            not_local_dst = torch.arange(
                level_not_nodes.numel(),
                device=gate.device,
                dtype=torch.long,
            )
            schedule.append(
                (
                    level_and_nodes,
                    torch.stack(
                        (
                            edge_src.index_select(0, and_positions),
                            and_local_dst,
                        ),
                        dim=0,
                    ),
                    level_not_nodes,
                    torch.stack(
                        (
                            edge_src.index_select(
                                0, level_not_start
                            ),
                            not_local_dst,
                        ),
                        dim=0,
                    ),
                )
            )
        return schedule, num_levels

    def load_pretrained(self, pretrained_model_path):
        self.load(pretrained_model_path)

    def forward(
        self,
        gate,
        edge_index,
        forward_level,
        forward_index,
        is_hs=False,
        batch=None,
        dg2_node_index=None,
        dg2_node_level=None,
        dg2_node_gate=None,
        dg2_edge_index=None,
        dg2_edge_level=None,
        dg2_edge_gate=None,
        _use_compact_schedule=True,
        _compact_schedule=None,
        _use_dag_state=True,
        _use_gru_cell=True,
    ):
        device = next(self.parameters()).device
        num_nodes = len(gate)
        gate_flat = gate.reshape(-1).to(device=device, dtype=torch.long)
        edge_index = edge_index.to(device=device, dtype=torch.long)
        forward_level = forward_level.reshape(-1).to(
            device=device, dtype=torch.long
        )
        forward_index = forward_index.reshape(-1).to(
            device=device, dtype=torch.long
        )
        if batch is None:
            batch = torch.zeros(num_nodes, device=device, dtype=torch.long)
        else:
            batch = batch.reshape(-1).to(device=device, dtype=torch.long)


        dtype = next(self.parameters()).dtype
        hs = torch.zeros(
            num_nodes, self.dim_hidden, device=device, dtype=dtype
        )
        pi_groups = _pi_groups(forward_level, forward_index, batch, gate_flat)
        if self.enable_encode:
            hs = generate_hs_init(
                forward_level,
                forward_index,
                hs,
                self.dim_hidden,
                batch,
                gate_flat,
                pi_groups=pi_groups,
            )
        hf = torch.zeros_like(hs)
        if self.pi_init_mode != 'const':
            hf = generate_hf_init(
                forward_level,
                forward_index,
                hf,
                self.dim_hidden,
                batch,
                pi_init_mode=self.pi_init_mode,
                gate=gate_flat,
                pi_groups=pi_groups,
            )


        const_nodes = torch.where(gate_flat == int(GateType.CONST))[0]
        if const_nodes.numel():
            hs[const_nodes] = 0.0
            hf[const_nodes] = self._const_function_anchor.to(dtype=dtype)

        node_state = torch.cat([hs, hf], dim=-1)

        if _use_compact_schedule:
            if _compact_schedule is None:
                compact_schedule, num_layers_f = (
                    self._compact_fanin_schedule(
                        gate_flat, edge_index, forward_level
                    )
                )
            else:
                compact_schedule, num_layers_f = _compact_schedule
        else:
            num_layers_f = int(torch.max(forward_level).item()) + 1
            edge_dst = edge_index[1]

        if _use_dag_state and _use_compact_schedule:
            node_state = self._propagate_dag(node_state, compact_schedule, use_gru_cell=_use_gru_cell)
            hs, hf = node_state.split(self.dim_hidden, dim=-1)
            return (hs, hf) if is_hs else hf

        for _ in range(self.num_rounds):
            for level in range(1, num_layers_f):
                if _use_compact_schedule:
                    (
                        l_and_node,
                        and_local_edge_index,
                        l_not_node,
                        not_local_edge_index,
                    ) = compact_schedule[level - 1]
                else:
                    layer_mask = forward_level == level
                    and_node_mask = layer_mask & (
                        gate_flat == int(GateType.AND)
                    )
                    l_and_node = forward_index[and_node_mask]
                    and_edge_mask = (
                        layer_mask.index_select(0, edge_dst)
                        & (
                            gate_flat.index_select(0, edge_dst)
                            == int(GateType.AND)
                        )
                    )
                    and_edge_index = edge_index[:, and_edge_mask]
                    not_node_mask = layer_mask & (
                        gate_flat == int(GateType.NOT)
                    )
                    l_not_node = forward_index[not_node_mask]
                    not_edge_mask = (
                        layer_mask.index_select(0, edge_dst)
                        & (
                            gate_flat.index_select(0, edge_dst)
                            == int(GateType.NOT)
                        )
                    )
                    not_edge_index = edge_index[:, not_edge_mask]
                if l_and_node.size(0) > 0:
                    if _use_compact_schedule:
                        and_msg = self.aggr_and_strc.forward_preindexed(
                            hs, and_local_edge_index, l_and_node
                        )
                    else:
                        and_msg = self.aggr_and_strc.forward_targets(
                            hs, and_edge_index, l_and_node
                        )
                    hs_and = torch.index_select(hs, dim=0, index=l_and_node)
                    _, hs_and = self.update_and_strc(and_msg.unsqueeze(0), hs_and.unsqueeze(0))


                    hs[l_and_node, :] = hs_and.squeeze(0).to(dtype=hs.dtype)


                    if _use_compact_schedule:
                        and_msg = self.aggr_and_func.forward_preindexed(
                            node_state,
                            and_local_edge_index,
                            l_and_node,
                        )
                    else:
                        and_msg = self.aggr_and_func.forward_targets(
                            node_state,
                            and_edge_index,
                            l_and_node,
                        )
                    hf_and = torch.index_select(hf, dim=0, index=l_and_node)
                    _, hf_and = self.update_and_func(and_msg.unsqueeze(0), hf_and.unsqueeze(0))
                    hf[l_and_node, :] = hf_and.squeeze(0).to(dtype=hf.dtype)

                if l_not_node.size(0) > 0:
                    if _use_compact_schedule:
                        not_msg = self.aggr_not_strc.forward_preindexed(
                            hs, not_local_edge_index, l_not_node
                        )
                    else:
                        not_msg = self.aggr_not_strc.forward_targets(
                            hs, not_edge_index, l_not_node
                        )
                    hs_not = torch.index_select(hs, dim=0, index=l_not_node)
                    _, hs_not = self.update_not_strc(not_msg.unsqueeze(0), hs_not.unsqueeze(0))
                    hs[l_not_node, :] = hs_not.squeeze(0).to(dtype=hs.dtype)


                    if _use_compact_schedule:
                        not_msg = self.aggr_not_func.forward_preindexed(
                            hf, not_local_edge_index, l_not_node
                        )
                    else:
                        not_msg = self.aggr_not_func.forward_targets(
                            hf, not_edge_index, l_not_node
                        )
                    hf_not = torch.index_select(hf, dim=0, index=l_not_node)
                    _, hf_not = self.update_not_func(not_msg.unsqueeze(0), hf_not.unsqueeze(0))
                    hf[l_not_node, :] = hf_not.squeeze(0).to(dtype=hf.dtype)


                node_state = torch.cat([hs, hf], dim=-1)


        node_embedding = node_state.squeeze(0)
        hs = node_embedding[:, :self.dim_hidden]
        hf = node_embedding[:, self.dim_hidden:]

        if is_hs:
            return hs, hf
        else:
            return hf

    def _propagate_dag(self, initial, schedule, *, use_gru_cell=True):
        state = DAGState(initial)
        dim = self.dim_hidden
        used_not = False
        for _ in range(self.num_rounds):
            for and_nodes, and_edges, not_nodes, not_edges in schedule:
                for nodes, edges, is_and in (
                    (and_nodes, and_edges, True),
                    (not_nodes, not_edges, False),
                ):
                    if not nodes.numel():
                        continue
                    source = state.read(edges[0])
                    target = state.read(nodes)
                    source_hs, source_hf = source.split(dim, dim=-1)
                    target_hs, target_hf = target.split(dim, dim=-1)
                    if is_and:
                        aggr_s, aggr_f = self.aggr_and_strc, self.aggr_and_func
                        update_s, update_f = self.update_and_strc, self.update_and_func
                        source_func, target_func = source, target
                    else:
                        used_not = True
                        aggr_s, aggr_f = self.aggr_not_strc, self.aggr_not_func
                        update_s, update_f = self.update_not_strc, self.update_not_func
                        source_func, target_func = source_hf, target_hf
                    if is_and:
                        msg_s = aggr_s.forward_local(source_hs, target_hs, edges[1])
                        msg_f = aggr_f.forward_local(source_func, target_func, edges[1])
                    else:
                        msg_s = aggr_s.update(aggr_s.msg_v(source_hs))
                        msg_f = aggr_f.update(aggr_f.msg_v(source_func))
                    if use_gru_cell:
                        new_hs = self._gru_step(update_s, msg_s, target_hs)
                        new_hf = self._gru_step(update_f, msg_f, target_hf)
                    else:
                        new_hs = update_s(msg_s.to(target_hs.dtype).unsqueeze(0), target_hs.unsqueeze(0))[1].squeeze(0)
                        new_hf = update_f(msg_f.to(target_hf.dtype).unsqueeze(0), target_hf.unsqueeze(0))[1].squeeze(0)
                    values = torch.cat((new_hs, new_hf), dim=-1)
                    state.write(nodes, values.to(dtype=initial.dtype))
        if used_not:
            zero = sum(
                parameter.reshape(-1)[0] * 0.0
                for aggr in (self.aggr_not_strc, self.aggr_not_func)
                for layer in (aggr.msg_q, aggr.msg_k, aggr.attn_lin)
                for parameter in layer.parameters()
            )
            state.dependency = state.dependency + zero
        return state.finish()

    @staticmethod
    def _gru_step(gru, message, hidden):
        mixed = message.is_cuda and torch.is_autocast_enabled("cuda")
        with torch.autocast(message.device.type, dtype=torch.float16, enabled=mixed):
            return torch._VF.gru_cell(
                message, hidden, gru.weight_ih_l0, gru.weight_hh_l0,
                gru.bias_ih_l0, gru.bias_hh_l0,
            )

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
        state_dict_ = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint


        state_dict = {}
        for k in state_dict_:
            if k.startswith('encoder.'):
                new_k = k[8:]

            elif k.startswith('module') and not k.startswith('module_list'):
                new_k = k[7:]
            else:
                new_k = k
            state_dict[new_k] = state_dict_[k]

        model_state_dict = self.state_dict()
        shape_mismatch = []
        extras = []
        filtered = {}
        for k, v in state_dict.items():
            if k in model_state_dict:
                if v.shape != model_state_dict[k].shape:
                    shape_mismatch.append((k, tuple(v.shape), tuple(model_state_dict[k].shape)))
                else:
                    filtered[k] = v
            else:
                extras.append(k)

        missing = [k for k in model_state_dict.keys() if k not in filtered]

        if extras:
            print(f'[DeepGate2.load] Ignoring {len(extras)} unexpected keys in checkpoint (e.g. {extras[:3]})')
        if shape_mismatch:
            raise RuntimeError(f'[DeepGate2.load] Shape mismatch: {shape_mismatch}')
        if missing:
            raise RuntimeError(
                f'[DeepGate2.load] {len(missing)} parameter(s) missing from checkpoint and would be randomly initialized: {missing[:5]}{"..." if len(missing) > 5 else ""}'
            )

        self.load_state_dict(filtered, strict=False)
