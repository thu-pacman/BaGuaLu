"""Apply bgl2 patches to Megatron-LM.

Importing this module is intentionally narrow: it only patches Megatron argument
registration/validation and the pipeline scheduler selector.
"""

from functools import wraps

from bgl2.adapter.fused_bias_swiglu import weighted_swiglu_backward, weighted_swiglu_forward
from bgl2.adapter.moe_imbalance import (
    moe_dispatch_wrapper,
    transformer_layer_forward_mlp_wrapper,
)
from bgl2.adapter.patch_utils import MegatronPatchesManager
from bgl2.arguments import chain_extra_args_provider
from bgl2.core.pipeline_parallel.selector import get_forward_backward_func_wrapper


def apply_numpy_compatibility_patches():
    import numpy as np

    if not hasattr(np, "product") and hasattr(np, "prod"):
        np.product = np.prod


def parse_args_wrapper(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        return parse_args(
            extra_args_provider=chain_extra_args_provider(extra_args_provider),
            ignore_unknown_args=ignore_unknown_args,
        )

    return wrapper


def validate_args_wrapper(validate_args):
    @wraps(validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}

        args = validate_args(args, defaults)
        backend = getattr(args, "pipeline_schedule_backend", "vanilla")
        fake_pp_warmup_enabled = (
            getattr(args, "bgl2_enable_pp_warmup_forward_backward", False)
            or getattr(args, "enable_pp_warmup_forward_backward", False)
        )

        if backend == "interleaved_1f1b":
            if args.pipeline_model_parallel_size <= 1:
                raise AssertionError(
                    "interleaved_1f1b requires --pipeline-model-parallel-size > 1"
                )
            if args.virtual_pipeline_model_parallel_size is None:
                raise AssertionError(
                    "interleaved_1f1b requires virtual pipeline parallelism"
                )

        if fake_pp_warmup_enabled:
            if backend != "interleaved_1f1b":
                raise AssertionError(
                    "--bgl2-enable-pp-warmup-forward-backward requires "
                    "--pipeline-schedule-backend interleaved_1f1b"
                )
            if args.pipeline_model_parallel_size <= 1:
                raise AssertionError(
                    "--bgl2-enable-pp-warmup-forward-backward requires "
                    "--pipeline-model-parallel-size > 1"
                )

        def arg(primary, fallback=None, default=None):
            if hasattr(args, primary):
                return getattr(args, primary)
            if fallback is not None and hasattr(args, fallback):
                return getattr(args, fallback)
            return default

        offload_requested = any(
            bool(arg(name))
            for name in ("bgl2_enable_activation_offload", "bgl2_patch_te")
        )

        if arg("bgl2_memory_max_vio", default=0.0) < 0:
            raise AssertionError("--bgl2-memory-max-vio must be non-negative")
        if arg("bgl2_memory_flash_attention_kv_threshold", default=192) < 1:
            raise AssertionError(
                "--bgl2-memory-flash-attention-kv-threshold must be >= 1"
            )
        if offload_requested:
            if arg("bgl2_offload_min_bytes", "offload_min_bytes", 1024 * 1024) < 0:
                raise AssertionError("--bgl2-offload-min-bytes must be non-negative")
            if not 0.0 <= arg("bgl2_activation_offload_ratio", "activation_offload_ratio", 1.0) <= 1.0:
                raise AssertionError("--bgl2-activation-offload-ratio must be in [0, 1]")
            threshold = arg("bgl2_activation_offload_threshold", "activation_offload_threshold", None)
            if threshold is not None and threshold < 0:
                raise AssertionError("--bgl2-activation-offload-threshold must be non-negative")
            if arg("bgl2_per_batch_offload_size", "per_batch_offload_size", 0) < 0:
                raise AssertionError("--bgl2-per-batch-offload-size must be non-negative")
            if arg("bgl2_activation_offload_stages", "activation_offload_stages", 1) < 1:
                raise AssertionError("--bgl2-activation-offload-stages must be >= 1")

        return args

    return wrapper


def training_log_wrapper(training_log):
    @wraps(training_log)
    def wrapper(*args, **kwargs):
        from megatron.training import get_args

        megatron_args = get_args()
        if not getattr(megatron_args, "report_memory_every_iteration", False):
            return training_log(*args, **kwargs)

        if "report_memory_flag" in kwargs:
            kwargs["report_memory_flag"] = True
            training_log(*args, **kwargs)
            return True

        args = list(args)
        if len(args) <= 6:
            raise TypeError("training_log missing positional report_memory_flag argument")
        args[6] = True
        training_log(*args, **kwargs)
        return True

    return wrapper


def apply_patches():
    apply_numpy_compatibility_patches()

    MegatronPatchesManager.register_patch(
        "megatron.training.arguments.parse_args",
        parse_args_wrapper,
        apply_wrapper=True,
    )
    MegatronPatchesManager.register_patch(
        "megatron.training.arguments.validate_args",
        validate_args_wrapper,
        apply_wrapper=True,
    )
    MegatronPatchesManager.register_patch(
        "megatron.core.pipeline_parallel.schedules.get_forward_backward_func",
        get_forward_backward_func_wrapper,
        apply_wrapper=True,
    )
    MegatronPatchesManager.register_patch(
        "megatron.training.training.training_log",
        training_log_wrapper,
        apply_wrapper=True,
    )
    MegatronPatchesManager.register_patch(
        "megatron.core.transformer.transformer_layer.TransformerLayer._forward_mlp",
        transformer_layer_forward_mlp_wrapper,
        apply_wrapper=True,
    )
    MegatronPatchesManager.register_patch(
        "megatron.core.transformer.moe.moe_layer.MoELayer.dispatch",
        moe_dispatch_wrapper,
        apply_wrapper=True,
    )
    MegatronPatchesManager.register_patch(
        "megatron.core.fusions.fused_bias_swiglu.WeightedSwiGLUFunction.forward",
        weighted_swiglu_forward,
    )
    MegatronPatchesManager.register_patch(
        "megatron.core.fusions.fused_bias_swiglu.WeightedSwiGLUFunction.backward",
        weighted_swiglu_backward,
    )


    args_holder = None

    def PatchDelegate(patched_func, te_func):
        def impl(*args, **kwargs):
            nonlocal args_holder
            if args_holder is None:
                from megatron.training import get_args
                args_holder = get_args()

            if any(
                getattr(args_holder, name, False)
                for name in ("bgl2_patch_te", "bgl2_enable_activation_offload")
            ):
                return patched_func(*args, **kwargs)
            return te_func(*args, **kwargs)
        return impl

    from patch_te import Linear, GroupedLinear, LayerNormLinear
    from transformer_engine.pytorch import (
        Linear as TELinear,
        GroupedLinear as TEGroupedLinear,
        LayerNormLinear as TELayerNormLinear,
    )

    MegatronPatchesManager.register_patch(
        'transformer_engine.pytorch.Linear.__init__',
        PatchDelegate(Linear.__init__, TELinear.__init__)
    )
    MegatronPatchesManager.register_patch(
        'transformer_engine.pytorch.GroupedLinear.__init__',
        PatchDelegate(GroupedLinear.__init__, TEGroupedLinear.__init__)
    )
    MegatronPatchesManager.register_patch(
        'transformer_engine.pytorch.LayerNormLinear.__init__',
        PatchDelegate(LayerNormLinear.__init__, TELayerNormLinear.__init__)
    )
    MegatronPatchesManager.register_patch(
        'transformer_engine.pytorch.Linear.forward',
        PatchDelegate(Linear.forward, TELinear.forward)
    )
    MegatronPatchesManager.register_patch(
        'transformer_engine.pytorch.GroupedLinear.forward',
        PatchDelegate(GroupedLinear.forward, TEGroupedLinear.forward)
    )
    MegatronPatchesManager.register_patch(
        'transformer_engine.pytorch.LayerNormLinear.forward',
        PatchDelegate(LayerNormLinear.forward, TELayerNormLinear.forward)
    )

    MegatronPatchesManager.apply_patches()


apply_patches()
