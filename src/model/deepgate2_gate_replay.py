import types
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from torch_geometric.utils import softmax, scatter


def gate_parameters(model, is_and):
    suffix = 'and' if is_and else 'not'
    result = []
    for channel in ('strc', 'func'):
        aggr = getattr(model, f'aggr_{suffix}_{channel}')
        assert aggr.msg_post is None
        layers = (aggr.msg_q, aggr.msg_k, aggr.msg_v, aggr.attn_lin) if is_and else (aggr.msg_v,)
        for layer in layers:
            result.extend((layer.weight, layer.bias))
    for channel in ('strc', 'func'):
        gru = getattr(model, f'update_{suffix}_{channel}')
        result.extend((gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0))
    return tuple(result)


def gate_operation(source, target, local_dst, parameters, is_and):
    dim = target.shape[-1] // 2
    source_hs, source_hf = source.split(dim, -1)
    target_hs, target_hf = target.split(dim, -1)
    width = 8 if is_and else 2

    def aggregate(x, y, p):
        if not is_and:
            return F.linear(x, p[0], p[1])
        q = F.linear(y.index_select(0, local_dst), p[0], p[1])
        k = F.linear(x, p[2], p[3])
        scores = F.linear(torch.cat((q, k), -1), p[6], p[7])
        probability = softmax(scores, local_dst, None, target.shape[0])
        messages = F.linear(x, p[4], p[5]) * probability
        return scatter(messages, local_dst, dim=-2, dim_size=target.shape[0], reduce='sum')

    msg_s = aggregate(source_hs, target_hs, parameters[:width])
    msg_f = aggregate(source if is_and else source_hf, target if is_and else target_hf, parameters[width:2*width])
    mixed = source.is_cuda and torch.is_autocast_enabled('cuda')
    with torch.autocast(source.device.type, dtype=torch.float16, enabled=mixed):
        hs = torch._VF.gru_cell(msg_s, target_hs, *parameters[2*width:2*width+4])
        hf = torch._VF.gru_cell(msg_f, target_hf, *parameters[2*width+4:2*width+8])
    return torch.cat((hs, hf), -1).to(target.dtype)


class GateReplay:
    def __init__(self, source, target, parameters, is_and, size):
        self.parameters = parameters
        self.capture_parameters = tuple(p.detach().requires_grad_() for p in parameters)
        self.arity = 2 if is_and else 1
        self.source = source.new_zeros((size*self.arity, source.shape[-1])).requires_grad_()
        self.target = target.new_zeros((size, target.shape[-1])).requires_grad_()
        self.local_dst = torch.arange(size, device=source.device).repeat_interleave(self.arity)
        self.dy = torch.zeros_like(self.target)
        mixed, dtype = torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda')

        def evaluate(backward=False):
            with torch.enable_grad():
                with torch.autocast('cuda', enabled=mixed, dtype=dtype, cache_enabled=False):
                    output = gate_operation(self.source, self.target, self.local_dst, self.capture_parameters, is_and)
                if backward:
                    with torch.autocast('cuda', enabled=False):
                        ds, dt, *dp = torch.autograd.grad(output, (self.source, self.target, *self.capture_parameters), self.dy)
                    return ds, dt, torch.cat([p.reshape(-1) for p in dp])
                return output.detach()

        torch.cuda.synchronize(source.device)
        stream = torch.cuda.Stream(device=source.device)
        stream.wait_stream(torch.cuda.current_stream(source.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                evaluate()
                evaluate(True)
        torch.cuda.current_stream(source.device).wait_stream(stream)
        torch.cuda.synchronize(source.device)
        self.forward_graph, self.backward_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.forward_graph, capture_error_mode='thread_local'):
            self.output = evaluate()
        with torch.cuda.graph(self.backward_graph, capture_error_mode='thread_local'):
            self.ds, self.dt, self.dp = evaluate(True)

    def forward(self, source, target):
        n = len(target)
        self.source[:len(source)].copy_(source)
        self.target[:n].copy_(target)
        self.forward_graph.replay()
        return self.output[:n].clone()

    def backward(self, source, target, gradient):
        n = len(target)
        self.source[:len(source)].copy_(source)
        self.target[:n].copy_(target)
        self.dy.zero_()
        self.dy[:n].copy_(gradient)
        self.backward_graph.replay()
        return self.ds[:len(source)].clone(), self.dt[:n].clone(), self.dp.clone()


class ReplayFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, replay, source, target, bank):
        ctx.replay = replay
        ctx.save_for_backward(source, target, bank, *replay.parameters)
        return replay.forward(source, target)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        source, target, *_ = ctx.saved_tensors
        ds, dt, dp = ctx.replay.backward(source, target, gradient)
        return None, ds, dt, dp


def prewarm(model, cache, max_bucket=128):
    parameter = next(model.parameters())
    device, dtype, dim = parameter.device, parameter.dtype, model.dim_hidden*2
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for is_and in (True, False):
            parameters = gate_parameters(model, is_and)
            for size in range(1, max_bucket+1):
                key = is_and, size, True, torch.bfloat16
                if key not in cache:
                    source = torch.zeros((size*(2 if is_and else 1), dim), device=device, dtype=dtype)
                    target = torch.zeros((size, dim), device=device, dtype=dtype)
                    cache[key] = GateReplay(source, target, parameters, is_and, size)


def install(model, max_bucket=128):
    from model.dag_state import DAGState
    old_propagate = model._propagate_dag
    cache = {}

    def propagate(self, initial, schedule, *, use_gru_cell=True):
        if not initial.is_cuda or not use_gru_cell:
            return old_propagate(initial, schedule, use_gru_cell=use_gru_cell)
        needs_backward = torch.is_grad_enabled()
        state = DAGState(initial)
        parameters = {kind: gate_parameters(self, kind) for kind in (True, False)}
        banks = {kind: torch.cat([p.reshape(-1) for p in ps]) for kind, ps in parameters.items()} if needs_backward else {}
        used_not = False
        for _ in range(self.num_rounds):
            for and_nodes, and_edges, not_nodes, not_edges in schedule:
                for nodes, edges, is_and in ((and_nodes, and_edges, True), (not_nodes, not_edges, False)):
                    n = len(nodes)
                    if not n:
                        continue
                    used_not |= not is_and
                    source, target = state.read(edges[0]), state.read(nodes)


                    size = n
                    if size <= max_bucket:
                        key = is_and, size, torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda')
                        if key not in cache:
                            cache[key] = GateReplay(source, target, parameters[is_and], is_and, size)
                        values = (ReplayFunction.apply(cache[key], source, target, banks[is_and])
                                  if needs_backward else cache[key].forward(source, target))
                    else:
                        values = gate_operation(source, target, edges[1], parameters[is_and], is_and)
                    state.write(nodes, values)
        if used_not and needs_backward:
            zero = sum(p.reshape(-1)[0]*0.0 for aggr in (self.aggr_not_strc,self.aggr_not_func)
                       for layer in (aggr.msg_q,aggr.msg_k,aggr.attn_lin) for p in layer.parameters())
            state.dependency = state.dependency + zero
        return state.finish()

    model._propagate_dag = types.MethodType(propagate, model)
    return cache
