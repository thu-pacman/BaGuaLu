"""Argument registration for the bgl2 pipeline scheduler extension."""


def _has_option(parser, option_string):
    return option_string in parser._option_string_actions


def add_bgl2_args(parser):
    """Add bgl2 scheduler/offload arguments to a Megatron-LM parser."""
    group = parser.add_argument_group(title="bgl2 pipeline scheduler")

    if not _has_option(parser, "--bgl2-enable-memory-modeling"):
        group.add_argument(
            "--bgl2-enable-memory-modeling",
            action="store_true",
            default=False,
            help=(
                "Enable analytical activation-memory modeling and MemoryReporter "
                "runtime output. Disabled by default."
            ),
        )

    if not _has_option(parser, "--bgl2-memory-max-vio"):
        group.add_argument(
            "--bgl2-memory-max-vio",
            type=float,
            default=0.0,
            help=(
                "MoE token imbalance ratio used by activation-memory modeling; "
                "modeled tokens are multiplied by 1 + ratio."
            ),
        )

    if not _has_option(parser, "--bgl2-memory-flash-attention-kv-threshold"):
        group.add_argument(
            "--bgl2-memory-flash-attention-kv-threshold",
            type=int,
            default=192,
            help="KV-channel threshold for automatic flash-attention modeling.",
        )

    if not _has_option(parser, "--pipeline-schedule-backend"):
        group.add_argument(
            "--pipeline-schedule-backend",
            "--bgl2-pipeline-schedule-backend",
            "--schedule-method",
            dest="pipeline_schedule_backend",
            default="vanilla",
            choices=["vanilla", "interleaved_1f1b"],
            help="Select Megatron vanilla scheduling or the bgl2 interleaved scheduler.",
        )

    if not _has_option(parser, "--bgl2-enable-activation-offload"):
        group.add_argument(
            "--bgl2-enable-activation-offload",
            action="store_true",
            default=False,
            help="Enable bgl2 scheduler-scoped activation offload/reload hooks.",
        )

    if not _has_option(parser, "--bgl2-offload-min-bytes"):
        group.add_argument(
            "--bgl2-offload-min-bytes",
            type=int,
            default=1024 * 1024,
            help="Minimum CUDA tensor size, in bytes, eligible for bgl2 activation offload.",
        )

    if not _has_option(parser, "--bgl2-offload-non-contiguous"):
        group.add_argument(
            "--bgl2-offload-non-contiguous",
            action="store_true",
            default=False,
            help=(
                "Allow bgl2 activation offload for non-contiguous or view-backed saved "
                "tensors. Disabled by default because some fused kernels require the "
                "original saved-tensor layout in backward."
            ),
        )

    if not _has_option(parser, "--bgl2-offload-granularity"):
        group.add_argument(
            "--bgl2-offload-granularity",
            choices=["microbatch", "layer"],
            default="microbatch",
            help=(
                "Offload eligible saved tensors after the complete scheduler microbatch "
                "or after each Transformer MLP layer. Layer mode reloads lazily during backward."
            ),
        )

    if not _has_option(parser, "--bgl2-no-offload-pin-memory"):
        group.add_argument(
            "--bgl2-no-offload-pin-memory",
            action="store_false",
            dest="bgl2_offload_pin_memory",
            default=True,
            help="Use regular CPU memory instead of pinned memory for bgl2 activation offload.",
        )

    if not _has_option(parser, "--bgl2-activation-offload-ratio") and not _has_option(
        parser, "--activation-offload-ratio"
    ):
        group.add_argument(
            "--bgl2-activation-offload-ratio",
            "--activation-offload-ratio",
            type=float,
            default=0.0,
            dest="bgl2_activation_offload_ratio",
            help="Fraction of eligible saved activations to offload for each scheduler microbatch.",
        )

    if not _has_option(parser, "--bgl2-activation-offload-threshold") and not _has_option(
        parser, "--activation-offload-threshold"
    ):
        group.add_argument(
            "--bgl2-activation-offload-threshold",
            "--activation-offload-threshold",
            type=float,
            default=None,
            dest="bgl2_activation_offload_threshold",
            help="Activation offload memory threshold in MiB.",
        )

    if not _has_option(parser, "--bgl2-per-batch-offload-size") and not _has_option(
        parser, "--per-batch-offload-size"
    ):
        group.add_argument(
            "--bgl2-per-batch-offload-size",
            "--per-batch-offload-size",
            type=float,
            default=0.0,
            dest="bgl2_per_batch_offload_size",
            help="Fixed per-microbatch activation offload budget in MiB; 0 disables batch-budget offload.",
        )

    if not _has_option(parser, "--bgl2-activation-offload-stages") and not _has_option(
        parser, "--activation-offload-stages"
    ):
        group.add_argument(
            "--bgl2-activation-offload-stages",
            "--activation-offload-stages",
            type=int,
            default=1,
            dest="bgl2_activation_offload_stages",
            help="Number of copy issue groups used by the bgl2 offload runtime.",
        )

    if not _has_option(parser, "--bgl2-prefetch-activation-reload"):
        group.add_argument(
            "--bgl2-prefetch-activation-reload",
            action="store_true",
            default=False,
            help=(
                "Reload the next interleaved microbatch during current compute. "
                "This improves overlap but keeps two restored activation groups live."
            ),
        )

    if not _has_option(parser, "--bgl2-offload-modules") and not _has_option(
        parser, "--offload-modules"
    ):
        group.add_argument(
            "--bgl2-offload-modules",
            "--offload-modules",
            nargs="*",
            default=["GroupedLinear", "LayerNormLinear", "swiglu", "permutation"],
            dest="bgl2_offload_modules",
            help="Patched op names whose saved tensors are eligible for bgl2 activation offload.",
        )

    if not _has_option(parser, "--force-fp8-ctx"):
        group.add_argument(
            "--force-fp8-ctx",
            action="store_true",
            default=False,
            help="Store selected saved activation contexts in FP8.",
        )

    if not _has_option(parser, "--fp8-ctx-modules"):
        group.add_argument(
            "--fp8-ctx-modules",
            nargs="*",
            type=str,
            default=None,
            help="Patched op names whose saved activation contexts use FP8 storage.",
        )

    pp_warmup_options = [
        option
        for option in (
            "--bgl2-enable-pp-warmup-forward-backward",
            "--enable-pp-warmup-forward-backward",
        )
        if not _has_option(parser, option)
    ]
    if pp_warmup_options:
        group.add_argument(
            *pp_warmup_options,
            dest="bgl2_enable_pp_warmup_forward_backward",
            action="store_true",
            default=False,
            help=(
                "Run one synthetic fake forward+backward before PP iteration 0. "
                "The fake pass uses generated dummy batches and discards gradients."
            ),
        )

    memory_report_options = [
        option
        for option in (
            "--report-memory-every-iteration",
            "--bgl2-report-memory-every-iteration",
        )
        if not _has_option(parser, option)
    ]
    if memory_report_options:
        group.add_argument(
            *memory_report_options,
            dest="report_memory_every_iteration",
            action="store_true",
            default=False,
            help=(
                "Keep Megatron's training memory report flag enabled after each "
                "training log call so memory usage is reported repeatedly."
            ),
        )

    if not _has_option(parser, "--bgl2-patch-te"):
        group.add_argument(
            "--bgl2-patch-te",
            action="store_true",
            default=False,
            dest="bgl2_patch_te",
            help="Use bgl2 patched Transformer Engine kernels.",
        )

    if not _has_option(parser, "--bgl2-export-moe-imbalance-ratio") and not _has_option(
        parser, "--export-moe-imbalance-ratio"
    ):
        group.add_argument(
            "--bgl2-export-moe-imbalance-ratio",
            "--export-moe-imbalance-ratio",
            action="store_true",
            default=False,
            dest="export_moe_imbalance_ratio",
            help=(
                "Export MoE token dispatch lifecycle events to stdout and imbalance "
                "metrics to stdout/TensorBoard."
            ),
        )

    return parser


def chain_extra_args_provider(extra_args_provider):
    """Return an extra-args provider that preserves the caller provider."""

    def provider(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        return add_bgl2_args(parser)

    return provider
