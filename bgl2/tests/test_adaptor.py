import argparse
import importlib
import sys
import types
import unittest


def _install_module(name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules[parent_name]
        setattr(parent, child_name, module)
    return module


def _install_fake_megatron():
    for name in list(sys.modules):
        if name == "megatron" or name.startswith("megatron."):
            del sys.modules[name]
    _install_fake_te_modules()

    _install_module("megatron")
    training_pkg = _install_module("megatron.training")
    arguments = _install_module("megatron.training.arguments")
    _install_module("megatron.core")
    _install_module("megatron.core.fusions")
    fused_bias_swiglu = _install_module("megatron.core.fusions.fused_bias_swiglu")
    _install_module("megatron.core.transformer")
    transformer_layer = _install_module("megatron.core.transformer.transformer_layer")
    _install_module("megatron.core.transformer.moe")
    moe_layer = _install_module("megatron.core.transformer.moe.moe_layer")
    pipeline_parallel = _install_module("megatron.core.pipeline_parallel")
    schedules = _install_module("megatron.core.pipeline_parallel.schedules")
    training = _install_module("megatron.training.training")

    state = types.SimpleNamespace(
        pipeline_schedule_backend="vanilla",
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        bgl2_enable_activation_offload=False,
        bgl2_enable_pp_warmup_forward_backward=False,
        bgl2_offload_min_bytes=1,
        bgl2_activation_offload_ratio=1.0,
        bgl2_per_batch_offload_size=0,
        bgl2_activation_offload_stages=1,
        bgl2_offload_modules=["GroupedLinear"],
        bgl2_patch_te=False,
        export_moe_imbalance_ratio=False,
        report_memory_every_iteration=False,
        curr_iteration=0,
    )

    def get_args():
        return state

    def parse_args(extra_args_provider=None, ignore_unknown_args=False):
        parser = argparse.ArgumentParser()
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        return parser.parse_args([])

    def validate_args(args, defaults=None):
        return args

    def original_get_forward_backward_func():
        return "vanilla-func"

    def training_log(*args, **kwargs):
        return args[6] if len(args) > 6 else kwargs["report_memory_flag"]

    training_pkg.get_args = get_args
    arguments.parse_args = parse_args
    arguments.validate_args = validate_args
    schedules.get_forward_backward_func = original_get_forward_backward_func
    pipeline_parallel.get_forward_backward_func = original_get_forward_backward_func
    training.get_forward_backward_func = original_get_forward_backward_func
    training.training_log = training_log

    class WeightedSwiGLUFunction:
        @staticmethod
        def forward(ctx, input, weights, fp8_input_store):
            return "original-forward"

        @staticmethod
        def backward(ctx, grad_output):
            return "original-backward"

    def weighted_swiglu(input, weights):
        return ("weighted", input, weights)

    def weighted_swiglu_back(grad_output, input, weights):
        return ("tmp", grad_output, input), ("wgrad", weights)

    fused_bias_swiglu.WeightedSwiGLUFunction = WeightedSwiGLUFunction
    fused_bias_swiglu.weighted_swiglu = weighted_swiglu
    fused_bias_swiglu.weighted_swiglu_back = weighted_swiglu_back

    class MoELayer:
        layer_number = 7

        def router_and_preprocess(self, hidden_states):
            return "local-tokens", "probs", hidden_states

        def dispatch(self, hidden_states, probs):
            return "dispatched-tokens", probs

    class TransformerLayer:
        def __init__(self):
            self.current_microbatch = -1
            self.mlp = MoELayer()

        def _forward_mlp(self, hidden_states, inference_context=None):
            return self.mlp.router_and_preprocess(hidden_states)

    moe_layer.MoELayer = MoELayer
    transformer_layer.TransformerLayer = TransformerLayer

    return state, arguments, schedules, pipeline_parallel, training


def _install_fake_te_modules():
    for name in list(sys.modules):
        if (
            name == "patch_te"
            or name == "transformer_engine"
            or name.startswith("transformer_engine.")
        ):
            del sys.modules[name]

    def make_linear_class(class_name, backend):
        class FakeLinear:
            def __init__(self):
                self.backend = backend

            def forward(self):
                return backend

        FakeLinear.__name__ = class_name
        return FakeLinear

    patch_te = types.ModuleType("patch_te")
    patch_te.Linear = make_linear_class("Linear", "patch_te")
    patch_te.GroupedLinear = make_linear_class("GroupedLinear", "patch_te")
    patch_te.LayerNormLinear = make_linear_class("LayerNormLinear", "patch_te")
    sys.modules["patch_te"] = patch_te

    _install_module("transformer_engine")
    pytorch = _install_module("transformer_engine.pytorch")
    pytorch.Linear = make_linear_class("TELinear", "te")
    pytorch.GroupedLinear = make_linear_class("TEGroupedLinear", "te")
    pytorch.LayerNormLinear = make_linear_class("TELayerNormLinear", "te")
    return patch_te, pytorch


class Bgl2AdaptorTest(unittest.TestCase):
    def test_bgl2_import_patches_args_and_scheduler_aliases(self):
        state, arguments, schedules, pipeline_parallel, training = _install_fake_megatron()

        fake_bgl2_schedules = types.ModuleType("bgl2.core.pipeline_parallel.schedules")

        def forward_backward_pipelining_with_interleaving():
            return "interleaved-func"

        fake_bgl2_schedules.forward_backward_pipelining_with_interleaving = (
            forward_backward_pipelining_with_interleaving
        )
        sys.modules["bgl2.core.pipeline_parallel.schedules"] = fake_bgl2_schedules

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        importlib.import_module("bgl2.adapter.megatron_adaptor")

        args = arguments.parse_args()
        self.assertTrue(hasattr(args, "pipeline_schedule_backend"))
        self.assertEqual(schedules.get_forward_backward_func(), "vanilla-func")
        self.assertIs(pipeline_parallel.get_forward_backward_func, schedules.get_forward_backward_func)
        self.assertIs(training.get_forward_backward_func, schedules.get_forward_backward_func)

        state.pipeline_schedule_backend = "interleaved_1f1b"
        selected = schedules.get_forward_backward_func()
        self.assertIs(selected, forward_backward_pipelining_with_interleaving)

    def test_bgl2_megatron_adaptor_entrypoint_imports_patch_module(self):
        _install_fake_megatron()

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        sys.modules.pop("bgl2.megatron_adaptor", None)

        module = importlib.import_module("bgl2.megatron_adaptor")

        self.assertTrue(hasattr(module, "apply_patches"))

    def test_bgl2_package_import_exposes_megatron_adaptor(self):
        _install_fake_megatron()

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        sys.modules.pop("bgl2.megatron_adaptor", None)

        from bgl2 import megatron_adaptor

        self.assertTrue(hasattr(megatron_adaptor, "apply_patches"))

    def test_vanilla_backend_uses_bgl2_scheduler_when_offload_is_enabled(self):
        state, arguments, schedules, pipeline_parallel, training = _install_fake_megatron()

        fake_bgl2_schedules = types.ModuleType("bgl2.core.pipeline_parallel.schedules")

        def get_forward_backward_func():
            return "bgl2-vanilla-func"

        fake_bgl2_schedules.get_forward_backward_func = get_forward_backward_func
        sys.modules["bgl2.core.pipeline_parallel.schedules"] = fake_bgl2_schedules

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        importlib.import_module("bgl2.adapter.megatron_adaptor")

        state.pipeline_schedule_backend = "vanilla"
        state.bgl2_enable_activation_offload = True

        self.assertEqual(schedules.get_forward_backward_func(), "bgl2-vanilla-func")
        self.assertIs(pipeline_parallel.get_forward_backward_func, schedules.get_forward_backward_func)
        self.assertIs(training.get_forward_backward_func, schedules.get_forward_backward_func)

    def test_vanilla_backend_uses_bgl2_scheduler_for_patch_te(self):
        state, arguments, schedules, pipeline_parallel, training = _install_fake_megatron()

        fake_bgl2_schedules = types.ModuleType("bgl2.core.pipeline_parallel.schedules")

        def get_forward_backward_func():
            return "bgl2-vanilla-func"

        fake_bgl2_schedules.get_forward_backward_func = get_forward_backward_func
        sys.modules["bgl2.core.pipeline_parallel.schedules"] = fake_bgl2_schedules

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        importlib.import_module("bgl2.adapter.megatron_adaptor")

        state.pipeline_schedule_backend = "vanilla"
        state.bgl2_patch_te = True

        self.assertEqual(schedules.get_forward_backward_func(), "bgl2-vanilla-func")
        self.assertIs(pipeline_parallel.get_forward_backward_func, schedules.get_forward_backward_func)
        self.assertIs(training.get_forward_backward_func, schedules.get_forward_backward_func)

    def test_numpy_product_alias_is_restored_for_numpy_2(self):
        import numpy as np

        had_product = hasattr(np, "product")
        original_product = getattr(np, "product", None)

        try:
            if had_product:
                delattr(np, "product")

            _install_fake_megatron()

            from bgl2.adapter.patch_utils import MegatronPatchesManager

            MegatronPatchesManager.reset()
            sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
            importlib.import_module("bgl2.adapter.megatron_adaptor")

            self.assertIs(np.product, np.prod)
            self.assertEqual(np.product([2, 3, 4]), 24)
        finally:
            if had_product:
                np.product = original_product
            elif hasattr(np, "product"):
                delattr(np, "product")

    def test_non_contiguous_offload_is_opt_in(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())

        args = parser.parse_args([])
        self.assertFalse(args.bgl2_offload_non_contiguous)

        args = parser.parse_args(["--bgl2-offload-non-contiguous"])
        self.assertTrue(args.bgl2_offload_non_contiguous)

    def test_activation_reload_prefetch_is_opt_in(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())

        args = parser.parse_args([])
        self.assertFalse(args.bgl2_prefetch_activation_reload)

        args = parser.parse_args(["--bgl2-prefetch-activation-reload"])
        self.assertTrue(args.bgl2_prefetch_activation_reload)

    def test_offload_granularity_defaults_to_microbatch_and_accepts_layer(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())

        args = parser.parse_args([])
        self.assertEqual(args.bgl2_offload_granularity, "microbatch")

        args = parser.parse_args(["--bgl2-offload-granularity", "layer"])
        self.assertEqual(args.bgl2_offload_granularity, "layer")

    def test_offload_modules_default_to_patched_op_names(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())

        args = parser.parse_args([])

        self.assertIn("GroupedLinear", args.bgl2_offload_modules)
        self.assertIn("LayerNormLinear", args.bgl2_offload_modules)
        self.assertIn("swiglu", args.bgl2_offload_modules)
        self.assertEqual(args.bgl2_activation_offload_ratio, 0.0)
        self.assertEqual(args.bgl2_per_batch_offload_size, 0.0)

    def test_fp8_activation_context_arguments_match_dcu_megatron(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())

        defaults = parser.parse_args([])
        configured = parser.parse_args(
            ["--force-fp8-ctx", "--fp8-ctx-modules", "GroupedLinear", "swiglu"]
        )

        self.assertFalse(defaults.force_fp8_ctx)
        self.assertIsNone(defaults.fp8_ctx_modules)
        self.assertTrue(configured.force_fp8_ctx)
        self.assertEqual(configured.fp8_ctx_modules, ["GroupedLinear", "swiglu"])

    def test_activation_memory_modeling_arguments_are_configurable(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())
        defaults = parser.parse_args([])
        self.assertFalse(defaults.bgl2_enable_memory_modeling)
        self.assertEqual(defaults.bgl2_memory_max_vio, 0.0)
        self.assertEqual(defaults.bgl2_memory_flash_attention_kv_threshold, 192)

        configured = parser.parse_args(
            [
                "--bgl2-enable-memory-modeling",
                "--bgl2-memory-max-vio",
                "0.5",
            ]
        )
        self.assertTrue(configured.bgl2_enable_memory_modeling)
        self.assertEqual(configured.bgl2_memory_max_vio, 0.5)

    def test_offload_argument_aliases_populate_bgl2_fields(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())

        args = parser.parse_args(
            [
                "--bgl2-patch-te",
                "--activation-offload-ratio",
                "0.25",
                "--activation-offload-threshold",
                "4096",
                "--per-batch-offload-size",
                "128",
                "--activation-offload-stages",
                "4",
                "--offload-modules",
                "GroupedLinear",
                "swiglu",
            ]
        )

        self.assertTrue(args.bgl2_patch_te)
        self.assertEqual(args.bgl2_activation_offload_ratio, 0.25)
        self.assertEqual(args.bgl2_activation_offload_threshold, 4096)
        self.assertEqual(args.bgl2_per_batch_offload_size, 128)
        self.assertEqual(args.bgl2_activation_offload_stages, 4)
        self.assertEqual(args.bgl2_offload_modules, ["GroupedLinear", "swiglu"])

    def test_patch_manager_patches_nested_class_attribute(self):
        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        for name in list(sys.modules):
            if name == "transformer_engine" or name.startswith("transformer_engine."):
                del sys.modules[name]

        try:
            _install_module("transformer_engine")
            pytorch = _install_module("transformer_engine.pytorch")

            class Linear:
                def __init__(self):
                    self.backend = "original"

            def patched_init(self):
                self.backend = "patched"

            pytorch.Linear = Linear

            MegatronPatchesManager.register_patch(
                "transformer_engine.pytorch.Linear.__init__",
                patched_init,
            )
            MegatronPatchesManager.apply_patches()

            self.assertEqual(pytorch.Linear().backend, "patched")
            self.assertIs(pytorch.Linear.__init__, patched_init)
        finally:
            MegatronPatchesManager.reset()
            for name in list(sys.modules):
                if name == "transformer_engine" or name.startswith("transformer_engine."):
                    del sys.modules[name]

    def test_fake_pp_warmup_argument_aliases_are_registered(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())

        args = parser.parse_args([])
        self.assertFalse(args.bgl2_enable_pp_warmup_forward_backward)

        args = parser.parse_args(["--bgl2-enable-pp-warmup-forward-backward"])
        self.assertTrue(args.bgl2_enable_pp_warmup_forward_backward)

        args = parser.parse_args(["--enable-pp-warmup-forward-backward"])
        self.assertTrue(args.bgl2_enable_pp_warmup_forward_backward)

    def test_fake_pp_warmup_validation_requires_interleaved_pp(self):
        state, arguments, _, _, _ = _install_fake_megatron()

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        importlib.import_module("bgl2.adapter.megatron_adaptor")

        state.bgl2_enable_pp_warmup_forward_backward = True
        state.pipeline_model_parallel_size = 2
        state.pipeline_schedule_backend = "vanilla"
        with self.assertRaisesRegex(AssertionError, "interleaved_1f1b"):
            arguments.validate_args(state)

        state.pipeline_schedule_backend = "interleaved_1f1b"
        state.virtual_pipeline_model_parallel_size = 2
        state.pipeline_model_parallel_size = 1
        with self.assertRaisesRegex(AssertionError, "pipeline-model-parallel-size > 1"):
            arguments.validate_args(state)

    def test_report_memory_every_iteration_argument_aliases_are_registered(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())

        args = parser.parse_args([])
        self.assertFalse(args.report_memory_every_iteration)

        args = parser.parse_args(["--report-memory-every-iteration"])
        self.assertTrue(args.report_memory_every_iteration)

        args = parser.parse_args(["--bgl2-report-memory-every-iteration"])
        self.assertTrue(args.report_memory_every_iteration)

    def test_moe_imbalance_argument_aliases_are_registered(self):
        from bgl2.arguments import add_bgl2_args

        parser = add_bgl2_args(argparse.ArgumentParser())

        args = parser.parse_args([])
        self.assertFalse(args.export_moe_imbalance_ratio)

        args = parser.parse_args(["--export-moe-imbalance-ratio"])
        self.assertTrue(args.export_moe_imbalance_ratio)

        args = parser.parse_args(["--bgl2-export-moe-imbalance-ratio"])
        self.assertTrue(args.export_moe_imbalance_ratio)

    def test_report_memory_every_iteration_keeps_training_log_flag_enabled(self):
        state, _, _, _, training = _install_fake_megatron()
        calls = []

        def training_log(*args, **kwargs):
            report_memory_flag = args[6] if len(args) > 6 else kwargs["report_memory_flag"]
            calls.append(report_memory_flag)
            return False

        training.training_log = training_log

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        importlib.import_module("bgl2.adapter.megatron_adaptor")

        base_args = [None, None, None, None, 1, None, False, False, None, None, None]
        self.assertFalse(training.training_log(*base_args))
        self.assertEqual(calls[-1], False)

        state.report_memory_every_iteration = True
        self.assertTrue(training.training_log(*base_args))
        self.assertEqual(calls[-1], True)

    def test_moe_imbalance_dispatch_patch_is_gated_by_argument(self):
        state, _, _, _, _ = _install_fake_megatron()

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        importlib.import_module("bgl2.adapter.megatron_adaptor")

        from bgl2.adapter import moe_imbalance

        calls = []
        original_export = moe_imbalance._export_moe_imbalance_ratio
        moe_layer = sys.modules["megatron.core.transformer.moe.moe_layer"]

        def export(layer, output, args, **kwargs):
            calls.append((layer, output, args, kwargs))

        try:
            moe_imbalance._export_moe_imbalance_ratio = export
            layer = moe_layer.MoELayer()

            output = layer.dispatch("local-tokens", "probs")
            self.assertEqual(output, ("dispatched-tokens", "probs"))
            self.assertEqual(calls, [])

            state.export_moe_imbalance_ratio = True
            output = layer.dispatch("local-tokens", "probs")
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], layer)
            self.assertEqual(calls[0][1], output)
            self.assertIs(calls[0][2], state)
            self.assertIn("local_splits", calls[0][3])
        finally:
            moe_imbalance._export_moe_imbalance_ratio = original_export

    def test_moe_dispatch_updates_dynamic_offload_without_metrics(self):
        state, _, _, _, _ = _install_fake_megatron()

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        importlib.import_module("bgl2.adapter.megatron_adaptor")

        from bgl2.adapter import moe_imbalance

        calls = []
        original_observe = moe_imbalance._observe_activation_demand
        moe_layer = sys.modules["megatron.core.transformer.moe.moe_layer"]

        try:
            moe_imbalance._observe_activation_demand = (
                lambda layer, output, args: calls.append((layer, output, args))
            )
            state.export_moe_imbalance_ratio = False
            state.bgl2_activation_offload_threshold = 4096
            layer = moe_layer.MoELayer()

            output = layer.dispatch("local-tokens", "probs")

            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0][0], layer)
            self.assertEqual(calls[0][1], output)
            self.assertIs(calls[0][2], state)
        finally:
            moe_imbalance._observe_activation_demand = original_observe

    def test_transformer_layer_mlp_patch_propagates_current_microbatch(self):
        _install_fake_megatron()

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        importlib.import_module("bgl2.adapter.megatron_adaptor")

        transformer_layer = sys.modules["megatron.core.transformer.transformer_layer"]
        layer = transformer_layer.TransformerLayer()

        events = []

        class FakeScope:
            def __init__(self, layer_number):
                self.layer_number = layer_number

            def __enter__(self):
                events.append(("scope-enter", self.layer_number))

            def __exit__(self, *exc_info):
                events.append(("scope-exit", self.layer_number))

        fake_offload = types.ModuleType("bgl2.core.pipeline_parallel.offload")
        fake_offload.mlp_activation_offload_scope = FakeScope
        original_offload = sys.modules.get("bgl2.core.pipeline_parallel.offload")
        sys.modules["bgl2.core.pipeline_parallel.offload"] = fake_offload

        try:
            layer.current_microbatch = 3
            self.assertEqual(layer._forward_mlp("hidden"), ("local-tokens", "probs", "hidden"))
            self.assertEqual(layer.mlp.current_microbatch, 3)
            self.assertEqual(events, [("scope-enter", 7), ("scope-exit", 7)])
        finally:
            if original_offload is None:
                sys.modules.pop("bgl2.core.pipeline_parallel.offload", None)
            else:
                sys.modules["bgl2.core.pipeline_parallel.offload"] = original_offload

    def test_megatron_adaptor_patches_weighted_swiglu_function(self):
        _install_fake_megatron()

        from bgl2.adapter.patch_utils import MegatronPatchesManager

        MegatronPatchesManager.reset()
        sys.modules.pop("bgl2.adapter.megatron_adaptor", None)
        importlib.import_module("bgl2.adapter.megatron_adaptor")
        from bgl2.adapter import fused_bias_swiglu as adapter_swiglu

        fused_bias_swiglu = sys.modules["megatron.core.fusions.fused_bias_swiglu"]

        self.assertIs(
            fused_bias_swiglu.WeightedSwiGLUFunction.forward,
            adapter_swiglu.weighted_swiglu_forward,
        )
        self.assertIs(
            fused_bias_swiglu.WeightedSwiGLUFunction.backward,
            adapter_swiglu.weighted_swiglu_backward,
        )

        offload = types.ModuleType("bgl2.core.pipeline_parallel.offload")
        packs = []

        def pack_hook(tensor, op_name=None):
            packs.append((tensor, op_name))
            return ("packed", tensor)

        def unpack_hook(tensor_pack):
            return tensor_pack[1]

        offload.pack_hook = pack_hook
        offload.unpack_hook = unpack_hook
        original_offload = sys.modules.get("bgl2.core.pipeline_parallel.offload")
        sys.modules["bgl2.core.pipeline_parallel.offload"] = offload

        try:
            ctx = types.SimpleNamespace()

            def save_for_backward(*tensors):
                ctx.saved_tensors = tensors

            ctx.save_for_backward = save_for_backward
            input_tensor = types.SimpleNamespace(dtype="bf16")

            output = fused_bias_swiglu.WeightedSwiGLUFunction.forward(
                ctx, input_tensor, "weights", False
            )
            grads = fused_bias_swiglu.WeightedSwiGLUFunction.backward(ctx, "grad")

            self.assertEqual(output, ("weighted", input_tensor, "weights"))
            self.assertEqual(packs, [(input_tensor, "swiglu")])
            self.assertEqual(ctx.saved_tensors, ("weights",))
            self.assertEqual(grads, (("tmp", "grad", input_tensor), ("wgrad", "weights"), None))
        finally:
            if original_offload is None:
                sys.modules.pop("bgl2.core.pipeline_parallel.offload", None)
            else:
                sys.modules["bgl2.core.pipeline_parallel.offload"] = original_offload


if __name__ == "__main__":
    unittest.main()
