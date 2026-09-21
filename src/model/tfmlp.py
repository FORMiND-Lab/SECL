import torch
from torch_geometric.nn import MessagePassing
import torch.nn as nn

from typing import Optional
from torch import Tensor
from torch_geometric.utils import softmax
from torch_geometric.typing import Adj, OptTensor
from .mlp import MLP


class TFMlpAggr(MessagePassing):
    def __init__(self, in_channels, ouput_channels=64, reverse=False, mlp_post=None):
        super(TFMlpAggr, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')
        if ouput_channels is None:
            ouput_channels = in_channels
        assert (in_channels > 0) and (ouput_channels > 0), 'The dimension for the DeepSetConv should be larger than 0.'

        self.msg_post = None if mlp_post is None else mlp_post
        self.attn_lin = nn.Linear(ouput_channels + ouput_channels, 1)

        self.msg_q = nn.Linear(in_channels, ouput_channels)
        self.msg_k = nn.Linear(in_channels, ouput_channels)
        self.msg_v = nn.Linear(in_channels, ouput_channels)

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def forward_targets(self, x, edge_index, target_nodes, edge_attr=None):
        if target_nodes.numel() == 0:
            return x.new_empty((0, self.msg_v.out_features))
        if self.flow != 'source_to_target':
            raise ValueError("forward_targets only supports source_to_target flow")
        local_target = x.index_select(0, target_nodes)
        local_dst = torch.searchsorted(target_nodes, edge_index[1])
        local_edge_index = torch.stack((edge_index[0], local_dst), dim=0)
        return self.propagate(
            local_edge_index,
            x=(x, local_target),
            edge_attr=edge_attr,
            size=(x.size(0), target_nodes.numel()),
        )

    def forward_fanins(
        self, x, source_nodes, target_nodes, fanin_count
    ):
        if target_nodes.numel() == 0:
            return x.new_empty((0, self.msg_v.out_features))
        if self.flow != 'source_to_target':
            raise ValueError("forward_fanins only supports source_to_target flow")
        if source_nodes.numel() != target_nodes.numel() * fanin_count:
            raise ValueError("source_nodes do not match the fixed fanin count")
        local_dst = torch.arange(
            target_nodes.numel(),
            device=target_nodes.device,
            dtype=torch.long,
        ).repeat_interleave(fanin_count)
        local_edge_index = torch.stack((source_nodes, local_dst), dim=0)
        return self.forward_preindexed(
            x, local_edge_index, target_nodes
        )

    def forward_preindexed(self, x, local_edge_index, target_nodes):
        if target_nodes.numel() == 0:
            return x.new_empty((0, self.msg_v.out_features))
        if self.flow != 'source_to_target':
            raise ValueError(
                "forward_preindexed only supports source_to_target flow"
            )
        local_target = x.index_select(0, target_nodes)
        return self.propagate(
            local_edge_index,
            x=(x, local_target),
            edge_attr=None,
            size=(x.size(0), target_nodes.numel()),
        )

    def message(self, x_i, x_j, edge_attr: Optional[Tensor], index: Tensor, ptr: OptTensor, size_i: Optional[int]):
        h_attn_q_i = self.msg_q(x_i)
        h_attn = self.msg_k(x_j)

        a_j = self.attn_lin(torch.cat([h_attn_q_i, h_attn], dim=-1))
        a_j = softmax(a_j, index, ptr, size_i)

        t = self.msg_v(x_j) * a_j
        return t

    def forward_local(self, source, target, local_dst):
        messages = self.message(
            target.index_select(0, local_dst), source, None,
            local_dst, None, target.size(0),
        )
        return self.update(self.aggregate(messages, local_dst, dim_size=target.size(0)))

    def update(self, aggr_out):
        if self.msg_post is not None:
            return self.msg_post(aggr_out)
        else:
            return aggr_out
