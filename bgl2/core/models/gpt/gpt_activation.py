"""Analytical per-layer activation-memory model for MoE GPT training.

The tensor inventory follows the standalone bgl2 memory model while retaining
the API expected by the pipeline ``MemoryReporter``.  Each tensor has both a
semantic module name and zero or more patched-op names so total memory and
offload-eligible memory can be reported from the same model.
"""

from __future__ import annotations

from functools import reduce
from operator import mul


BYTES_PER_ELEMENT = {
    "fp64": 8,
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int64": 8,
    "int32": 4,
}
bytes_dict = BYTES_PER_ELEMENT
ATTENTION_TYPES = ("flash", "unfused")


def _arg(args, name, default=None):
    value = getattr(args, name, None)
    return default if value is None else value


def _memory_arg(args, name, default=None):
    value = getattr(args, f"bgl2_memory_{name}", None)
    if value is None:
        value = getattr(args, name, None)
    return default if value is None else value


def _runtime_value(args, transformer_config, name, default=None):
    if transformer_config is not None:
        value = getattr(transformer_config, name, None)
        if value is not None:
            return value
    return _arg(args, name, default)


def resolve_attention_type(attention_type=None, args=None, transformer_config=None):
    if args is None:
        from megatron.training import get_args

        args = get_args()

    if attention_type is None:
        backend = str(
            _runtime_value(args, transformer_config, "attention_backend", "")
        ).lower()
        backend_name = backend.rsplit(".", 1)[-1]
        if backend_name in ("flash", "fused"):
            attention_type = "flash"
        elif backend_name in ("unfused", "local"):
            attention_type = "unfused"

    if attention_type is None:
        threshold = _memory_arg(args, "flash_attention_kv_threshold", 192)
        q_head_dim = _resolved_attention_q_head_dim(args, transformer_config)
        attention_type = "flash" if q_head_dim < threshold else "unfused"

    if attention_type not in ATTENTION_TYPES:
        raise ValueError(
            f"Unknown attention type {attention_type!r}; expected one of {ATTENTION_TYPES}"
        )
    return attention_type


def _resolved_kv_channels(args, transformer_config=None):
    kv_channels = _runtime_value(args, transformer_config, "kv_channels", None)
    if kv_channels is not None:
        return kv_channels
    hidden_size = _runtime_value(args, transformer_config, "hidden_size", 7168)
    attention_heads = _runtime_value(
        args, transformer_config, "num_attention_heads", 56
    )
    if hidden_size % attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    return hidden_size // attention_heads


def _resolved_query_groups(args, transformer_config=None):
    attention_heads = _runtime_value(
        args, transformer_config, "num_attention_heads", 56
    )
    if transformer_config is not None:
        query_groups = getattr(transformer_config, "num_query_groups", None)
        if query_groups is not None:
            return query_groups
    if not bool(
        _runtime_value(args, transformer_config, "group_query_attention", False)
    ):
        return attention_heads
    query_groups = _runtime_value(
        args, transformer_config, "num_query_groups", None
    )
    if query_groups is None:
        raise ValueError("num_query_groups is required when GQA is enabled")
    return query_groups


def _is_mla(args, transformer_config=None):
    return bool(
        _runtime_value(args, transformer_config, "multi_latent_attention", False)
    )


def _is_gated_linear_unit(args, transformer_config=None):
    gated_linear_unit = _runtime_value(
        args, transformer_config, "gated_linear_unit", None
    )
    if gated_linear_unit is not None:
        return bool(gated_linear_unit)
    return bool(_arg(args, "swiglu", False))


def _required_runtime_value(args, transformer_config, name):
    value = _runtime_value(args, transformer_config, name, None)
    if value is None:
        raise ValueError(f"{name} is required when multi_latent_attention is enabled")
    return value


def _resolved_attention_q_head_dim(args, transformer_config=None):
    if not _is_mla(args, transformer_config):
        return _resolved_kv_channels(args, transformer_config)
    return _required_runtime_value(
        args, transformer_config, "qk_head_dim"
    ) + _required_runtime_value(
        args, transformer_config, "qk_pos_emb_head_dim"
    )


def _resolved_recompute_modules(args):
    granularity = getattr(args, "recompute_granularity", None)
    legacy_enabled = bool(getattr(args, "recompute_activations", False))
    if not granularity and not legacy_enabled:
        return set(), False
    if granularity == "full":
        return set(), True
    modules = getattr(args, "recompute_modules", None)
    return set(modules or ("core_attn",)), False


def _dtype_name(value):
    if value is None:
        return None
    name = str(value).lower()
    if "bfloat16" in name or name == "bf16":
        return "bf16"
    if "float16" in name or "half" in name or name == "fp16":
        return "fp16"
    if "float64" in name or "double" in name or name == "fp64":
        return "fp64"
    if "float32" in name or name in ("float", "fp32"):
        return "fp32"
    return None


def _compute_dtype(args, transformer_config=None):
    dtype = _dtype_name(
        _runtime_value(args, transformer_config, "params_dtype", None)
    )
    if dtype is not None:
        return dtype
    if bool(getattr(args, "bf16", False)):
        return "bf16"
    if bool(getattr(args, "fp16", False)):
        return "fp16"
    return "fp32"


def _activation_dtypes(args, transformer_config=None):
    compute_dtype = _compute_dtype(args, transformer_config)
    fp8_enabled = _runtime_value(args, transformer_config, "fp8", None) is not None
    quantized_dtype = "fp8" if fp8_enabled else compute_dtype
    softmax_dtype = "fp32"
    router_score_dtype = (
        _runtime_value(args, transformer_config, "moe_router_dtype", None)
        or compute_dtype
    )
    topk_index_dtype = "int64"
    permutation_index_dtype = (
        "int32"
        if bool(_runtime_value(args, transformer_config, "moe_permute_fusion", False))
        else "int64"
    )
    activation_input_dtype = (
        "fp8"
        if fp8_enabled
        and bool(
            _runtime_value(
                args,
                transformer_config,
                "activation_func_fp8_input_store",
                False,
            )
        )
        else compute_dtype
    )
    for dtype in (
        compute_dtype,
        quantized_dtype,
        softmax_dtype,
        router_score_dtype,
        topk_index_dtype,
        permutation_index_dtype,
        activation_input_dtype,
    ):
        if dtype not in BYTES_PER_ELEMENT:
            raise ValueError(f"Unsupported activation dtype: {dtype}")
    return (
        compute_dtype,
        quantized_dtype,
        softmax_dtype,
        router_score_dtype,
        topk_index_dtype,
        permutation_index_dtype,
        activation_input_dtype,
    )


class Module:
    """One retained activation tensor."""

    def __init__(
        self,
        name,
        description,
        shape,
        layerid,
        dtype="bf16",
        recompute=False,
        offload=0.0,
        params=None,
        offload_names=(),
    ):
        if dtype not in BYTES_PER_ELEMENT:
            raise ValueError(f"Unsupported activation dtype: {dtype}")
        self.name = name
        self.description = description
        self.shape = None if shape is not None and len(shape) == 0 else shape
        self.layerid = layerid
        self.dtype = dtype
        self.recompute = recompute
        self.offload = offload
        self.offload_names = set(offload_names)
        self.filled_shape = None
        self._params = None
        self.bytes_per_element = BYTES_PER_ELEMENT[dtype]
        if params is not None:
            self.set_values(params)

    def set_values(self, params):
        if params is None:
            raise ValueError("Activation shape parameters must be provided")
        if self.shape is None:
            return
        self._params = dict(params)
        filled_shape = []
        for dimension in self.shape:
            if isinstance(dimension, (int, float)):
                filled_shape.append(dimension)
            elif "/" in dimension:
                numerator, denominator = dimension.split("/", 1)
                filled_shape.append(params[numerator] / params[denominator])
            else:
                filled_shape.append(params[dimension])
        self.filled_shape = filled_shape

    def matches(self, modules):
        if modules is None:
            return True
        selected = set(modules)
        return self.name in selected or bool(self.offload_names & selected)

    def set_recompute(self, recompute_modules):
        if self.name in recompute_modules:
            self.recompute = True

    def set_offload(self, offload_modules, ratio):
        if self.matches(offload_modules):
            self.offload = ratio

    def set_dynamic_value(self, param_name, value):
        if self.shape is None:
            return
        if self._params is None:
            raise RuntimeError("Activation tensor values must be initialized first")
        self._params[f"dynamic_{param_name}"] = value
        self.set_values(self._params)

    def display(self):
        print(f"Layer {self.layerid}: {self.name} ({self.description})")
        print(
            f"  Shape: {self.shape}\tRecompute: {self.recompute}\t"
            f"Offload ops: {sorted(self.offload_names)}"
        )
        if self.filled_shape is not None:
            print(
                f"  Actual Shape: {self.filled_shape}\tdtype: {self.dtype}\t"
                f"Size: {self.calculate() / 1024**2:.2f} MiB"
            )

    def calculate(self, modules=None):
        if not self.matches(modules) or self.shape is None or self.recompute:
            return 0
        if self.filled_shape is None:
            raise RuntimeError(
                f"Activation shape for {self.description!r} is not initialized"
            )
        elements = reduce(mul, self.filled_shape, 1)
        return int(elements * self.bytes_per_element * (1.0 - self.offload))


class Model:
    def __init__(self, name, modules, metadata=None):
        self.name = name
        self.modules = modules
        self.metadata = metadata or {}
        self._params = None

    def set_values(self, params):
        self._params = dict(params)
        for module in self.modules:
            module.set_values(params)

    def set_recompute(self, recompute_modules):
        for module in self.modules:
            module.set_recompute(recompute_modules)

    def set_offload(self, offload_modules, ratio):
        for module in self.modules:
            module.set_offload(offload_modules, ratio)

    def set_dynamic_value(self, param_name, value):
        if self._params is None:
            raise RuntimeError("Activation model values must be initialized first")
        self._params[f"dynamic_{param_name}"] = value
        self.set_values(self._params)

    def display(self):
        print(f"Model {self.name}: {self.metadata}")
        for module in self.modules:
            module.display()

    def calculate(self, selected_modules=None, except_modules=None):
        if selected_modules is not None and except_modules is not None:
            raise ValueError("selected_modules and except_modules are mutually exclusive")
        if selected_modules is not None:
            return sum(module.calculate(selected_modules) for module in self.modules)
        total = sum(module.calculate() for module in self.modules)
        if except_modules is not None:
            return total - sum(module.calculate(except_modules) for module in self.modules)
        return total


def args_to_params(args=None, max_vio=None, transformer_config=None):
    if args is None:
        from megatron.training import get_args

        args = get_args()
    if max_vio is None:
        max_vio = float(_memory_arg(args, "max_vio", 0.0))
    batch_size = _runtime_value(args, transformer_config, "micro_batch_size", 1)
    seq_len = _runtime_value(args, transformer_config, "seq_length", 4096)
    topk = _runtime_value(args, transformer_config, "moe_router_topk", 1)
    moe_hidden_size = _runtime_value(
        args, transformer_config, "moe_ffn_hidden_size", None
    )
    if moe_hidden_size is None:
        moe_hidden_size = _runtime_value(
            args, transformer_config, "ffn_hidden_size", 2048
        )
    hidden_size = _runtime_value(args, transformer_config, "hidden_size", 7168)
    attention_heads = _runtime_value(
        args, transformer_config, "num_attention_heads", 56
    )
    tp = _runtime_value(
        args, transformer_config, "tensor_model_parallel_size", 1
    )
    mla = _is_mla(args, transformer_config)

    params = {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tp": tp,
        "etp": _runtime_value(
            args, transformer_config, "expert_tensor_parallel_size", 1
        ),
        "hidden_size": hidden_size,
        "attention_head": attention_heads,
        "topk": topk,
        "expert_num": _runtime_value(args, transformer_config, "num_experts", 1),
        "moe_hidden_size": moe_hidden_size,
        "moe_fc1_projection_size": moe_hidden_size
        * (2 if _is_gated_linear_unit(args, transformer_config) else 1),
        "dynamic_ntoken": (1.0 + max_vio) * batch_size * seq_len * topk,
    }

    if mla:
        qk_head_dim = _required_runtime_value(
            args, transformer_config, "qk_head_dim"
        )
        qk_pos_emb_head_dim = _required_runtime_value(
            args, transformer_config, "qk_pos_emb_head_dim"
        )
        v_head_dim = _required_runtime_value(
            args, transformer_config, "v_head_dim"
        )
        kv_lora_rank = _required_runtime_value(
            args, transformer_config, "kv_lora_rank"
        )
        q_lora_rank = _runtime_value(
            args, transformer_config, "q_lora_rank", None
        )
        params.update(
            {
                "q_down_projection_size": q_lora_rank or 0,
                "kv_down_projection_size": kv_lora_rank
                + qk_pos_emb_head_dim,
                "q_up_projection_size": attention_heads
                * (qk_head_dim + qk_pos_emb_head_dim),
                "kv_up_projection_size": attention_heads
                * (qk_head_dim + v_head_dim),
                "attention_kv_head": attention_heads,
                "attention_q_head_dim": qk_head_dim + qk_pos_emb_head_dim,
                "attention_v_head_dim": v_head_dim,
            }
        )
    else:
        query_groups = _resolved_query_groups(args, transformer_config)
        kv_channels = _resolved_kv_channels(args, transformer_config)
        params.update(
            {
                "qkv_projection_size": (attention_heads + 2 * query_groups)
                * kv_channels,
                "attention_kv_head": query_groups,
                "attention_q_head_dim": kv_channels,
                "attention_v_head_dim": kv_channels,
            }
        )
    return params


class MoETransformer(Model):
    def __init__(self, args, transformer_config=None):
        (
            compute,
            quantized,
            softmax,
            router_score,
            topk_index,
            permutation_index,
            activation_input,
        ) = _activation_dtypes(args, transformer_config)
        attention_type = resolve_attention_type(
            args=args, transformer_config=transformer_config
        )
        modules = [
            Module("input_layernorm", "rmsnorm (output)", ["batch_size", "seq_len", "hidden_size/tp"], 1, quantized, offload_names={"LayerNormLinear"}),
        ]
        if _is_mla(args, transformer_config):
            qk_layernorm = bool(
                _runtime_value(args, transformer_config, "qk_layernorm", False)
            )
            up_projection_offload_names = (
                {"LayerNormLinear"} if qk_layernorm else {"Linear"}
            )
            if _runtime_value(args, transformer_config, "q_lora_rank", None) is not None:
                modules.append(
                    Module("mla_q_down_proj", "q down projection (output)", ["batch_size", "seq_len", "q_down_projection_size/tp"], 2, compute, offload_names={"Linear"})
                )
            modules.extend(
                [
                    Module("mla_kv_down_proj", "kv down projection (output)", ["batch_size", "seq_len", "kv_down_projection_size/tp"], 2, compute, offload_names={"Linear"}),
                    Module("mla_q_up_proj", "q up projection (output)", ["batch_size", "seq_len", "q_up_projection_size/tp"], 2, compute, offload_names=up_projection_offload_names),
                    Module("mla_kv_up_proj", "kv up projection (output)", ["batch_size", "seq_len", "kv_up_projection_size/tp"], 2, compute, offload_names=up_projection_offload_names),
                ]
            )
        else:
            modules.append(
                Module("linear_qkv", "gemm (output)", ["batch_size", "seq_len", "qkv_projection_size/tp"], 2, compute, offload_names={"LayerNormLinear"})
            )
        modules.extend(
            [
                Module("rotary_pos_emb", "q (output)", ["batch_size", "seq_len", "attention_head/tp", "attention_q_head_dim"], 3, compute),
                Module("rotary_pos_emb", "k (output)", ["batch_size", "seq_len", "attention_kv_head/tp", "attention_q_head_dim"], 3, compute),
            ]
        )
        if attention_type == "flash":
            modules.extend(
                [
                    Module("core_attention", "attention (output)", ["batch_size", "seq_len", "attention_head/tp", "attention_v_head_dim"], 4, compute, offload_names={"core_attn"}),
                    Module("flash_attn", "buffer", ["batch_size", "seq_len", "attention_head/tp", "attention_q_head_dim"], 4, compute, offload_names={"core_attn"}),
                    Module("core_attention", "softmax_lse", ["batch_size", "seq_len", "attention_head/tp"], 4, softmax, offload_names={"core_attn"}),
                ]
            )
        else:
            modules.extend(
                [
                    Module("pre_core_attn", "v (output)", ["batch_size", "seq_len", "attention_kv_head/tp", "attention_v_head_dim"], 3, compute, offload_names={"core_attn"}),
                    Module("core_attention", "mul qk", ["batch_size", "seq_len", "seq_len", "attention_head/tp"], 4, compute, offload_names={"core_attn"}),
                    Module("attn_proj", "attention output", ["batch_size", "seq_len", "attention_head/tp", "attention_v_head_dim"], 5, compute, offload_names={"core_attn"}),
                ]
            )
        modules.extend(
            [
                Module("attn_proj", "projection output", ["batch_size", "seq_len", "hidden_size/tp"], 5, quantized, offload_names={"Linear"}),
                Module("self_attn", "bda", ["batch_size", "seq_len", "hidden_size/tp"], 6, compute),
                Module("pre_mlp_norm", "rmsnorm (output)", ["batch_size", "seq_len", "hidden_size/tp"], 7, compute, offload_names={"LayerNormLinear"}),
                Module("moe_router", "topk_scores", ["batch_size", "seq_len", "topk"], 8, router_score),
                Module("moe_router", "topk_indices", ["batch_size", "seq_len", "topk"], 8, topk_index),
                Module("moe_router", "scores (output)", ["batch_size", "seq_len", "expert_num"], 8, router_score),
                Module("moe_router", "sorted_indices (output)", ["batch_size", "seq_len", "expert_num"], 8, permutation_index),
                Module("moe_fc1", "inputmats", ["dynamic_ntoken", "etp/tp", "hidden_size"], 9, quantized, offload_names={"GroupedLinear", "permutation"}),
            ]
        )
        if _is_gated_linear_unit(args, transformer_config):
            modules.extend(
                [
                    Module("moe_fc1", "gated projection output", ["dynamic_ntoken/tp", "moe_fc1_projection_size"], 10, activation_input, offload_names={"swiglu"}),
                    Module("moe_fc1", "gated activation output", ["dynamic_ntoken/tp", "moe_hidden_size"], 11, quantized, offload_names={"swiglu", "GroupedLinear"}),
                ]
            )
        else:
            modules.extend(
                [
                    Module("moe_fc1", "output", ["dynamic_ntoken/tp", "moe_hidden_size"], 10, compute),
                    Module("moe_fc1", "gelu (output)", ["dynamic_ntoken/tp", "moe_hidden_size"], 11, quantized, offload_names={"GroupedLinear"}),
                ]
            )
        modules.append(
            Module("mlp_bda", "bda", ["batch_size", "seq_len/tp", "hidden_size"], 12, compute)
        )

        recompute_modules, full_recompute = _resolved_recompute_modules(args)
        recompute_map = {
            "core_attn": {"core_attention"},
            "moe_act": {"mlp_bda"},
            "layernorm": {"input_layernorm", "pre_mlp_norm"},
            "mla_up_proj": {"mla_q_up_proj", "mla_kv_up_proj", "rotary_pos_emb"},
            "mlp": {"mlp"},
            "moe": {"moe_fc1"},
            "experts": {"moe_fc1", "moe_fc2"},
            "router": {"moe_router"},
        }
        recomputed_names = set()
        for recompute_module in recompute_modules:
            recomputed_names.update(recompute_map.get(recompute_module, ()))
        for module in modules:
            module.recompute = full_recompute or module.name in recomputed_names

        super().__init__(
            "moe_transformer",
            modules,
            metadata={
                "attention_type": attention_type,
                "attention_mode": "mla" if _is_mla(args, transformer_config) else "qkv",
                "gated_linear_unit": _is_gated_linear_unit(args, transformer_config),
                "compute_dtype": compute,
                "quantized_dtype": quantized,
                "router_score_dtype": router_score,
                "topk_index_dtype": topk_index,
                "permutation_index_dtype": permutation_index,
                "activation_input_dtype": activation_input,
                "max_vio": float(_memory_arg(args, "max_vio", 0.0)),
            },
        )


# Compatibility constructor for callers introduced by the first migration.
class DSV3(MoETransformer):
    def __init__(self, args=None, transformer_config=None):
        if args is None:
            from megatron.training import get_args

            args = get_args()
        super().__init__(args, transformer_config=transformer_config)


dsv3 = DSV3


activation_model = None


def set_activation_model(model="moe_transformer", args=None, transformer_config=None):
    global activation_model
    if model not in ("moe_transformer", "dsv3"):
        raise ValueError(f"Unknown activation model: {model}")
    if args is None:
        from megatron.training import get_args

        args = get_args()
    activation_model = MoETransformer(args, transformer_config=transformer_config)
    return activation_model


def get_activation_model():
    if activation_model is None:
        raise RuntimeError(
            "Activation model is not set. Call initialize_activation_model() first."
        )
    return activation_model


def initialize_activation_model(
    model="moe_transformer", args=None, transformer_config=None
):
    configured_model = set_activation_model(
        model, args=args, transformer_config=transformer_config
    )
    configured_model.set_values(
        args_to_params(args, transformer_config=transformer_config)
    )
    return configured_model


def _offload_modules(args):
    modules = getattr(args, "offload_modules", None)
    if modules is None:
        modules = getattr(args, "bgl2_offload_modules", None)
    return [] if modules is None else modules


def calculate_memory_consumption(args=None):
    if args is None:
        from megatron.training import get_args

        args = get_args()
    return get_activation_model().calculate(selected_modules=_offload_modules(args))
