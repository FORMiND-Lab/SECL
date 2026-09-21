import numpy as np
import torch


def return_forward_order_info(edge_index, num_nodes):
    edges = torch.as_tensor(edge_index).detach().cpu().to(torch.long)
    if edges.dim() != 2 or edges.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, E], got {tuple(edges.shape)}"
        )
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if edges.numel() and (
        bool((edges < 0).any()) or int(edges.max()) >= num_nodes
    ):
        raise ValueError("edge_index contains an out-of-range node ID")

    forward_index = torch.arange(num_nodes, dtype=torch.long)
    if num_nodes == 0:
        return torch.empty(0, dtype=torch.long), forward_index

    source = edges[0].contiguous().numpy()
    target = edges[1].contiguous().numpy()
    in_degree = np.bincount(target, minlength=num_nodes).astype(
        np.int64, copy=False
    )
    out_count = np.bincount(source, minlength=num_nodes).astype(
        np.int64, copy=False
    )
    offsets = np.empty(num_nodes + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(out_count, out=offsets[1:])
    if source.size:
        edge_order = np.argsort(source, kind="stable")
        sorted_target = target[edge_order]
    else:
        sorted_target = target

    level = np.full(num_nodes, -1, dtype=np.int64)
    queue = np.empty(num_nodes, dtype=np.int64)
    roots = np.flatnonzero(in_degree == 0)
    queue[: roots.size] = roots
    level[roots] = 0
    head = 0
    tail = int(roots.size)
    while head < tail:
        node = int(queue[head])
        head += 1
        next_level = level[node] + 1
        for successor in sorted_target[offsets[node] : offsets[node + 1]]:
            successor = int(successor)
            if level[successor] < next_level:
                level[successor] = next_level
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                queue[tail] = successor
                tail += 1
    if tail != num_nodes:
        raise ValueError("edge_index is not a DAG")
    return torch.from_numpy(level), forward_index
