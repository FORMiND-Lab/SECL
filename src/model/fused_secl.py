from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable
from .loss import bounded_tanh_multiplier


def _forward(queries, keys, positive, positions, temperature):
    raw = queries @ keys.t()
    work_dtype = torch.float64 if queries.dtype == torch.float64 else torch.float32
    cosine = raw.to(work_dtype).clamp(-1.0 + 1e-4, 1.0 - 1e-4)
    mask = torch.arange(keys.size(0), device=keys.device)[None, :] == positions[:, None]
    logits = torch.where(mask, positive[:, None], cosine.square() / temperature)
    return torch.logsumexp(logits, dim=1)


def _backward(queries, keys, positive, positions, normalizer, gradient, temperature, control):
    raw = queries @ keys.t()
    work_dtype = torch.float64 if queries.dtype == torch.float64 else torch.float32
    raw_work = raw.to(work_dtype)
    cosine = raw_work.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
    mask = torch.arange(keys.size(0), device=keys.device)[None, :] == positions[:, None]
    logits = torch.where(mask, positive[:, None], cosine.square() / temperature)
    probability_gradient = (logits - normalizer[:, None]).exp() * gradient[:, None]
    positive_gradient = torch.where(mask, probability_gradient, 0.0).sum(dim=1)
    log_m, delta, lower, upper = control.unbind()
    multiplier = bounded_tanh_multiplier(cosine, False, log_m, delta, lower, upper)


    derivative = (probability_gradient / temperature) * (2.0 * cosine) * multiplier
    active = (~mask) & (raw_work >= -1.0 + 1e-4) & (raw_work <= 1.0 - 1e-4)
    cosine_gradient = torch.where(active, derivative, 0.0).to(raw.dtype)
    query_gradient = (cosine_gradient @ keys).to(queries.dtype)
    key_gradient = (cosine_gradient.t() @ queries).to(keys.dtype)
    return query_gradient, key_gradient, positive_gradient.to(positive.dtype)


@lru_cache(maxsize=1)
def _kernels():
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    return (
        torch.compile(_forward, dynamic=True, fullgraph=True),
        torch.compile(_backward, dynamic=True, fullgraph=True),
    )


@lru_cache(maxsize=64)
def _control_tensor(log_m, delta, lower, upper, device, dtype):
    return torch.tensor([log_m, delta, lower, upper], device=device, dtype=dtype)


class _Denominator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, keys, positive, positions, temperature, log_m, delta, lower, upper):
        ctx.temperature = temperature
        ctx.mixed = torch.is_autocast_enabled("cuda")
        ctx.mixed_dtype = torch.get_autocast_dtype("cuda")
        result = _kernels()[0](queries, keys, positive, positions, temperature)
        dtype = torch.float64 if queries.dtype == torch.float64 else torch.float32
        control = _control_tensor(log_m, delta, lower, upper, queries.device, dtype)
        ctx.save_for_backward(queries, keys, positive, positions, result, control)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        queries, keys, positive, positions, normalizer, control = ctx.saved_tensors
        with torch.autocast("cuda", enabled=ctx.mixed, dtype=ctx.mixed_dtype):
            dq, dk, dp = _kernels()[1](
                queries, keys, positive, positions, normalizer, gradient,
                ctx.temperature, control,
            )
        return dq, dk, dp, None, None, None, None, None, None


def denominator(queries, keys, positive, positions, temperature, log_m, delta, lower, upper):
    return _Denominator.apply(queries, keys, positive, positions, temperature, log_m, delta, lower, upper)


import triton
import triton.language as tl


@triton.jit
def _circuit_denominator(Q, K, P, Pos, Tiles, Z, DY, DQ,
                         D: tl.constexpr, BD: tl.constexpr, B: tl.constexpr,
                         T: tl.constexpr, LOG_M: tl.constexpr, DELTA: tl.constexpr,
                         LOWER: tl.constexpr, UPPER: tl.constexpr, BACK: tl.constexpr):
    tile = tl.program_id(0)
    start, end = tl.load(Tiles+tile*4), tl.load(Tiles+tile*4+1)
    key_start, key_end = tl.load(Tiles+tile*4+2), tl.load(Tiles+tile*4+3)
    rows, channels = start+tl.arange(0, B), tl.arange(0, BD)
    q = tl.load(Q+rows[:, None]*D+channels[None, :], (rows[:, None] < end) & (channels[None, :] < D), 0)
    positions = tl.load(Pos+rows, rows < end, -1)+key_start
    positive = tl.load(P+rows, rows < end, 0)
    if BACK:
        normalizer = tl.load(Z+rows, rows < end, 0)
        dy = tl.load(DY+rows, rows < end, 0)
        total = tl.full((B, BD), 0., tl.float32)
    else:
        maximum = tl.full((B,), -float('inf'), tl.float32)
        total = tl.full((B,), 0., tl.float32)
    for first in range(key_start, key_end, B):
        columns = first+tl.arange(0, B)
        keys = tl.load(K+channels[:, None]+columns[None, :]*D,
                       (channels[:, None] < D) & (columns[None, :] < key_end), 0)
        raw = tl.dot(q, keys).to(q.dtype).to(tl.float32)
        cosine = tl.minimum(tl.maximum(raw, -1.+1.e-4), 1.-1.e-4)
        own = positions[:, None] == columns[None, :]
        logits = tl.where(own, positive[:, None], cosine*cosine/T)
        valid = (rows[:, None] < end) & (columns[None, :] < key_end)
        if BACK:
            probability = tl.exp(logits-normalizer[:, None])*dy[:, None]
            multiplier = tl.exp(LOG_M*(2./(1.+tl.exp(-2.*((tl.abs(cosine)-.8)/DELTA)))-1.))
            multiplier = tl.minimum(tl.maximum(multiplier, LOWER), UPPER)
            derivative = (probability/T)*(2.*cosine)*multiplier
            derivative = tl.where(valid & ~own & (raw >= -1.+1.e-4) & (raw <= 1.-1.e-4), derivative, 0.)
            total += tl.dot(derivative.to(q.dtype), tl.trans(keys))
        else:
            logits = tl.where(valid, logits, -float('inf'))
            updated = tl.maximum(maximum, tl.max(logits, 1))
            total = total*tl.exp(maximum-updated)+tl.sum(tl.exp(logits-updated[:, None]), 1)
            maximum = updated
    if BACK:
        tl.store(DQ+rows[:, None]*D+channels[None, :], total, (rows[:, None] < end) & (channels[None, :] < D))
    else:
        tl.store(Z+rows, maximum+tl.log(total), rows < end)


@triton.jit
def _circuit_key_gradient(Q, K, P, Pos, Tiles, Z, DY, DK,
                          D: tl.constexpr, BD: tl.constexpr, B: tl.constexpr,
                          T: tl.constexpr, LOG_M: tl.constexpr, DELTA: tl.constexpr,
                          LOWER: tl.constexpr, UPPER: tl.constexpr):
    tile = tl.program_id(0)
    first, end = tl.load(Tiles+tile*5), tl.load(Tiles+tile*5+1)
    query_start, query_end = tl.load(Tiles+tile*5+2), tl.load(Tiles+tile*5+3)
    key_start = tl.load(Tiles+tile*5+4)
    rows, channels = first+tl.arange(0, B), tl.arange(0, BD)
    keys = tl.load(K+rows[:, None]*D+channels[None, :], (rows[:, None] < end) & (channels[None, :] < D), 0)
    total = tl.full((B, BD), 0., tl.float32)
    for start in range(query_start, query_end, B):
        columns = start+tl.arange(0, B)
        queries = tl.load(Q+channels[:, None]+columns[None, :]*D,
                          (channels[:, None] < D) & (columns[None, :] < query_end), 0)
        raw = tl.dot(keys, queries).to(keys.dtype).to(tl.float32)
        cosine = tl.minimum(tl.maximum(raw, -1.+1.e-4), 1.-1.e-4)
        positive = tl.load(P+columns, columns < query_end, 0)
        positions = tl.load(Pos+columns, columns < query_end, -1)+key_start
        own = positions[None, :] == rows[:, None]
        logits = tl.where(own, positive[None, :], cosine*cosine/T)
        logz = tl.load(Z+columns, columns < query_end, 0)
        dy = tl.load(DY+columns, columns < query_end, 0)
        probability = tl.exp(logits-logz[None, :])*dy[None, :]
        multiplier = tl.exp(LOG_M*(2./(1.+tl.exp(-2.*((tl.abs(cosine)-.8)/DELTA)))-1.))
        multiplier = tl.minimum(tl.maximum(multiplier, LOWER), UPPER)
        derivative = (probability/T)*(2.*cosine)*multiplier
        valid = (rows[:, None] < end) & (columns[None, :] < query_end)
        derivative = tl.where(valid & ~own & (raw >= -1.+1.e-4) & (raw <= 1.-1.e-4), derivative, 0.)
        total += tl.dot(derivative.to(keys.dtype), tl.trans(queries))
    tl.store(DK+rows[:, None]*D+channels[None, :], total, (rows[:, None] < end) & (channels[None, :] < D))


class _CircuitDenominator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, queries, keys, positive, positions, query_tiles, key_tiles, temperature, log_m, delta, lower, upper):
        q, k = queries.to(torch.bfloat16), keys.to(torch.bfloat16)
        z = torch.empty(len(q), dtype=torch.float32, device=q.device)
        constants = (q.shape[1], triton.next_power_of_2(q.shape[1]), 64, temperature, log_m, delta, lower, upper)
        _circuit_denominator[(len(query_tiles),)](q, k, positive, positions, query_tiles, z, z, q,
                                                *constants, False, num_warps=4)
        ctx.save_for_backward(q, k, positive, positions, query_tiles, key_tiles, z)
        ctx.constants, ctx.dtypes = constants, (queries.dtype, keys.dtype)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        q, k, positive, positions, query_tiles, key_tiles, z = ctx.saved_tensors
        gradient = gradient.contiguous()
        dq, dk = torch.empty_like(q), torch.zeros_like(k)
        _circuit_denominator[(len(query_tiles),)](q, k, positive, positions, query_tiles, z, gradient, dq,
                                                *ctx.constants, True, num_warps=4)
        _circuit_key_gradient[(len(key_tiles),)](q, k, positive, positions, key_tiles, z, gradient, dk,
                                               *ctx.constants, num_warps=4)
        dp = (positive-z).exp()*gradient
        return dq.to(ctx.dtypes[0]), dk.to(ctx.dtypes[1]), dp, None, None, None, None, None, None, None, None


def circuit_denominator(queries, keys, positive, positions, query_tiles, key_tiles, temperature, log_m, delta, lower, upper):
    return _CircuitDenominator.apply(queries, keys, positive, positions, query_tiles, key_tiles,
                                     temperature, log_m, delta, lower, upper)
