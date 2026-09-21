import torch
from torch.autograd.function import once_differentiable


class _Read(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dependency, table, indices):
        ctx.table = table
        ctx.save_for_backward(indices)
        return table.values.index_select(0, indices)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        ctx.table.gradient.index_add_(0, indices, gradient)
        return gradient.new_zeros(()), None, None


class _Write(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dependency, values, table, indices):
        ctx.table = table
        ctx.save_for_backward(indices)
        table.values.index_copy_(0, indices, values)
        return dependency.new_zeros(())

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        values_gradient = ctx.table.gradient.index_select(0, indices)


        ctx.table.gradient.index_fill_(0, indices, 0)
        return gradient.new_zeros(()), values_gradient, None, None


class _Finish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dependency, table):
        ctx.table = table


        return table.values.view_as(table.values)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        ctx.table.gradient = gradient.clone()
        return gradient.new_zeros(()), None


class DAGState:
    def __init__(self, initial):
        if initial.requires_grad:
            raise ValueError("DAGState initial values must be constants")
        self.values = initial
        self.gradient = None
        self.dependency = initial.new_zeros((), requires_grad=True)

    def read(self, indices):
        return _Read.apply(self.dependency, self, indices)

    def write(self, indices, values):
        self.dependency = _Write.apply(self.dependency, values, self, indices)

    def finish(self):
        result = _Finish.apply(self.dependency, self)


        self.dependency = None
        return result
