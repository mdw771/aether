import torch


class TypeCastFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dtype):
        ctx.save_for_backward(x)
        ctx.original_dtype = x.dtype
        res = x.to(dtype)
        return res * 1.0

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output.to(ctx.original_dtype), None