"""Adapter patches for Megatron fused SwiGLU autograd functions."""


def weighted_swiglu_forward(ctx, input, weights, fp8_input_store):
    input_for_backward = input
    if fp8_input_store:
        import torch

        input_for_backward = input.to(torch.float8_e4m3fn)

    from bgl2.core.pipeline_parallel.offload import pack_hook
    from megatron.core.fusions import fused_bias_swiglu

    ctx.tensor_pack = pack_hook(input_for_backward, op_name="swiglu")
    ctx.save_for_backward(weights)
    ctx.ori_input_dtype = input.dtype
    ctx.fp8_input_store = fp8_input_store
    return fused_bias_swiglu.weighted_swiglu(input, weights)


def weighted_swiglu_backward(ctx, grad_output):
    from bgl2.core.pipeline_parallel.offload import unpack_hook
    from megatron.core.fusions import fused_bias_swiglu

    input = unpack_hook(ctx.tensor_pack)
    (weights,) = ctx.saved_tensors
    input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
    tmp, wgrad = fused_bias_swiglu.weighted_swiglu_back(grad_output, input, weights)
    return tmp, wgrad, None
