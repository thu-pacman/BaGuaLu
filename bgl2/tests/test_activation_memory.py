import types
import unittest

from bgl2.core.models.gpt import gpt_activation


def _args(**overrides):
    values = {
        "micro_batch_size": 2,
        "seq_length": 128,
        "tensor_model_parallel_size": 2,
        "expert_tensor_parallel_size": 1,
        "hidden_size": 256,
        "group_query_attention": True,
        "num_query_groups": 4,
        "kv_channels": 16,
        "num_attention_heads": 8,
        "multi_latent_attention": False,
        "num_experts": 8,
        "moe_router_topk": 2,
        "moe_ffn_hidden_size": 512,
        "ffn_hidden_size": 512,
        "swiglu": True,
        "params_dtype": "torch.bfloat16",
        "bf16": True,
        "fp16": False,
        "fp8": None,
        "moe_router_dtype": None,
        "moe_permute_fusion": True,
        "attention_backend": "AttnBackend.auto",
        "recompute_granularity": None,
        "recompute_modules": None,
        "bgl2_memory_max_vio": 0.0,
        "bgl2_offload_modules": [
            "GroupedLinear",
            "LayerNormLinear",
            "swiglu",
            "permutation",
        ],
    }
    values.update(overrides)
    return types.SimpleNamespace(**values)


class ActivationMemoryModelTest(unittest.TestCase):
    def test_selected_and_non_selected_groups_partition_total_memory(self):
        model = gpt_activation.initialize_activation_model(args=_args())
        selected = {"GroupedLinear", "LayerNormLinear", "swiglu", "permutation"}

        self.assertEqual(
            model.calculate(),
            model.calculate(selected_modules=selected)
            + model.calculate(except_modules=selected),
        )
        self.assertGreater(model.calculate(selected_modules=selected), 0)
        self.assertGreater(model.calculate(except_modules=selected), 0)

    def test_token_imbalance_increases_dynamic_but_not_static_memory(self):
        dynamic_groups = {"GroupedLinear", "permutation", "swiglu"}
        base = gpt_activation.initialize_activation_model(args=_args())
        imbalanced = gpt_activation.initialize_activation_model(
            args=_args(bgl2_memory_max_vio=1.0)
        )

        self.assertEqual(
            base.calculate(except_modules=dynamic_groups),
            imbalanced.calculate(except_modules=dynamic_groups),
        )
        self.assertGreater(
            imbalanced.calculate(selected_modules=dynamic_groups),
            base.calculate(selected_modules=dynamic_groups),
        )

    def test_runtime_token_observation_updates_dynamic_memory(self):
        model = gpt_activation.initialize_activation_model(args=_args())
        dynamic_groups = {"GroupedLinear", "permutation", "swiglu"}
        before = model.calculate(selected_modules=dynamic_groups)

        model.set_dynamic_value("ntoken", 1024)

        self.assertGreater(model.calculate(selected_modules=dynamic_groups), before)

    def test_bgl2_offload_argument_name_is_supported(self):
        args = _args(bgl2_offload_modules=["GroupedLinear"])
        model = gpt_activation.initialize_activation_model(args=args)

        self.assertEqual(
            gpt_activation.calculate_memory_consumption(args),
            model.calculate(selected_modules={"GroupedLinear"}),
        )

    def test_non_fp8_training_defaults_quantized_activations_to_compute_dtype(self):
        model = gpt_activation.initialize_activation_model(args=_args(fp8=None))
        self.assertEqual(model.metadata["compute_dtype"], "bf16")
        self.assertEqual(model.metadata["quantized_dtype"], "bf16")

        fp8_model = gpt_activation.initialize_activation_model(args=_args(fp8="hybrid"))
        self.assertEqual(fp8_model.metadata["quantized_dtype"], "fp8")
        self.assertLess(fp8_model.calculate(), model.calculate())

    def test_compute_and_router_dtypes_follow_megatron_arguments(self):
        model = gpt_activation.initialize_activation_model(
            args=_args(
                params_dtype="torch.float16",
                bf16=False,
                fp16=True,
                moe_router_dtype="fp32",
                moe_permute_fusion=False,
            )
        )

        self.assertEqual(model.metadata["compute_dtype"], "fp16")
        self.assertEqual(model.metadata["router_score_dtype"], "fp32")
        self.assertEqual(model.metadata["topk_index_dtype"], "int64")
        self.assertEqual(model.metadata["permutation_index_dtype"], "int64")

        fused = gpt_activation.initialize_activation_model(
            args=_args(moe_permute_fusion=True)
        )
        self.assertEqual(fused.metadata["permutation_index_dtype"], "int32")

    def test_attention_type_follows_megatron_backend(self):
        unfused = gpt_activation.initialize_activation_model(
            args=_args(attention_backend="AttnBackend.unfused")
        )
        fused = gpt_activation.initialize_activation_model(
            args=_args(attention_backend="AttnBackend.fused")
        )

        self.assertEqual(unfused.metadata["attention_type"], "unfused")
        self.assertEqual(fused.metadata["attention_type"], "flash")

    def test_qkv_projection_width_is_derived_for_gqa_and_mha(self):
        gqa = gpt_activation.initialize_activation_model(args=_args())
        gqa_qkv = next(module for module in gqa.modules if module.name == "linear_qkv")
        self.assertEqual(gqa._params["qkv_projection_size"], (8 + 2 * 4) * 16)
        self.assertEqual(gqa_qkv.filled_shape, [2, 128, 128.0])

        mha = gpt_activation.initialize_activation_model(
            args=_args(group_query_attention=False, num_query_groups=None)
        )
        mha_qkv = next(module for module in mha.modules if module.name == "linear_qkv")
        self.assertEqual(mha._params["qkv_projection_size"], (8 + 2 * 8) * 16)
        self.assertEqual(mha_qkv.filled_shape, [2, 128, 192.0])
        self.assertGreater(mha_qkv.calculate(), gqa_qkv.calculate())

    def test_transformer_config_attention_layout_takes_precedence(self):
        transformer_config = types.SimpleNamespace(
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=20,
        )
        model = gpt_activation.initialize_activation_model(
            args=_args(group_query_attention=False),
            transformer_config=transformer_config,
        )

        self.assertEqual(model._params["qkv_projection_size"], (8 + 2 * 2) * 20)

    def test_fc1_projection_width_follows_gated_linear_unit(self):
        gated = gpt_activation.initialize_activation_model(args=_args(swiglu=True))
        plain = gpt_activation.initialize_activation_model(args=_args(swiglu=False))
        gated_fc1 = next(
            module for module in gated.modules if module.layerid == 10
        )
        plain_fc1 = next(
            module for module in plain.modules if module.layerid == 10
        )

        self.assertEqual(gated._params["moe_fc1_projection_size"], 2 * 512)
        self.assertEqual(plain._params["moe_fc1_projection_size"], 512)
        self.assertEqual(gated_fc1.filled_shape, [256.0, 1024])
        self.assertEqual(plain_fc1.filled_shape, [256.0, 512])

    def test_transformer_config_gated_linear_unit_takes_precedence(self):
        gated = gpt_activation.initialize_activation_model(
            args=_args(swiglu=False),
            transformer_config=types.SimpleNamespace(gated_linear_unit=True),
        )
        plain = gpt_activation.initialize_activation_model(
            args=_args(swiglu=True),
            transformer_config=types.SimpleNamespace(gated_linear_unit=False),
        )

        self.assertTrue(gated.metadata["gated_linear_unit"])
        self.assertFalse(plain.metadata["gated_linear_unit"])
        self.assertEqual(gated._params["moe_fc1_projection_size"], 1024)
        self.assertEqual(plain._params["moe_fc1_projection_size"], 512)

    def test_mla_projection_widths_follow_transformer_config(self):
        transformer_config = types.SimpleNamespace(
            multi_latent_attention=True,
            num_attention_heads=8,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_head_dim=12,
            qk_pos_emb_head_dim=4,
            v_head_dim=10,
        )
        model = gpt_activation.initialize_activation_model(
            args=_args(), transformer_config=transformer_config
        )
        modules = {module.name: module for module in model.modules}

        self.assertEqual(model.metadata["attention_mode"], "mla")
        self.assertEqual(
            modules["mla_q_down_proj"].filled_shape,
            [2, 128, 16.0],
        )
        self.assertEqual(
            modules["mla_kv_down_proj"].filled_shape,
            [2, 128, 10.0],
        )
        self.assertEqual(
            modules["mla_q_up_proj"].filled_shape,
            [2, 128, 64.0],
        )
        self.assertEqual(
            modules["mla_kv_up_proj"].filled_shape,
            [2, 128, 88.0],
        )
        self.assertNotIn("linear_qkv", modules)

    def test_mla_without_q_lora_has_no_q_down_projection(self):
        transformer_config = types.SimpleNamespace(
            multi_latent_attention=True,
            num_attention_heads=8,
            q_lora_rank=None,
            kv_lora_rank=16,
            qk_head_dim=12,
            qk_pos_emb_head_dim=4,
            v_head_dim=10,
        )
        model = gpt_activation.initialize_activation_model(
            args=_args(), transformer_config=transformer_config
        )

        self.assertNotIn("mla_q_down_proj", {module.name for module in model.modules})

    def test_transformer_config_is_preferred_over_training_args(self):
        transformer_config = types.SimpleNamespace(
            params_dtype="torch.float16",
            fp8="hybrid",
            moe_router_dtype="fp64",
            moe_permute_fusion=False,
            attention_backend="AttnBackend.unfused",
            activation_func_fp8_input_store=True,
        )
        model = gpt_activation.initialize_activation_model(
            args=_args(), transformer_config=transformer_config
        )

        self.assertEqual(model.metadata["compute_dtype"], "fp16")
        self.assertEqual(model.metadata["quantized_dtype"], "fp8")
        self.assertEqual(model.metadata["router_score_dtype"], "fp64")
        self.assertEqual(model.metadata["permutation_index_dtype"], "int64")
        self.assertEqual(model.metadata["attention_type"], "unfused")
        self.assertEqual(model.metadata["activation_input_dtype"], "fp8")

    def test_recompute_reduces_retained_activation_memory(self):
        base = gpt_activation.initialize_activation_model(args=_args())
        recomputed = gpt_activation.initialize_activation_model(
            args=_args(
                recompute_granularity="selective",
                recompute_modules=["moe"],
            )
        )
        self.assertLess(recomputed.calculate(), base.calculate())

    def test_gqa_requires_query_groups(self):
        with self.assertRaisesRegex(ValueError, "num_query_groups"):
            gpt_activation.initialize_activation_model(
                args=_args(num_query_groups=None)
            )

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown activation model"):
            gpt_activation.set_activation_model("unknown", args=_args())


if __name__ == "__main__":
    unittest.main()
