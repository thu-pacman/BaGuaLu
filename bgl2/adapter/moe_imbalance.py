"""Monkey patches for exporting MoE token dispatch imbalance metrics."""

from functools import wraps
from itertools import count


UNKNOWN_FIELD = "unknown"
_MOE_TOKEN_EVENT_IDS = count()


def _get_megatron_args():
    try:
        from megatron.training import get_args

        return get_args()
    except Exception:
        return None


def _get_enabled_args(args=None):
    if args is None:
        args = _get_megatron_args()
    if args is None:
        return None

    if getattr(args, "export_moe_imbalance_ratio", False) or getattr(
        args, "bgl2_export_moe_imbalance_ratio", False
    ):
        return args
    return None


def _dynamic_offload_enabled(args):
    if args is None:
        return False
    threshold = getattr(args, "activation_offload_threshold", None)
    if threshold is None:
        threshold = getattr(args, "bgl2_activation_offload_threshold", None)
    return threshold is not None


def _rank_for_log():
    try:
        import torch

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    return 0


def _next_event_id():
    return next(_MOE_TOKEN_EVENT_IDS)


def _format_field(value):
    if value is None:
        return UNKNOWN_FIELD
    return str(value).replace(" ", "_")


def _first_existing_attr(obj, names, default=None):
    if obj is None:
        return default
    for name in names:
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if value is not None:
            return value
    return default


def _current_iteration(args):
    return _first_existing_attr(
        args,
        ("curr_iteration", "iteration", "global_step", "step"),
        default=0,
    )


def _current_microbatch(layer, args):
    value = _first_existing_attr(
        layer,
        ("current_microbatch", "microbatch", "microbatch_id"),
        default=None,
    )
    if value is not None and value != -1:
        return value

    value = _first_existing_attr(
        args,
        ("current_microbatch", "microbatch", "microbatch_id"),
        default=None,
    )
    if value is not None and value != -1:
        return value

    return UNKNOWN_FIELD


def _step_for_metrics(iteration):
    try:
        return int(iteration)
    except (TypeError, ValueError):
        return 0


def _is_dist_initialized(torch):
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _group_size(torch, group):
    if group is not None and hasattr(group, "size"):
        return group.size()
    return torch.distributed.get_world_size(group=group)


def _group_rank(torch, group):
    if group is not None and hasattr(group, "rank"):
        return group.rank()
    return torch.distributed.get_rank(group=group)


def _group_ranks(torch, group, world_size):
    if hasattr(torch.distributed, "get_process_group_ranks"):
        try:
            return list(torch.distributed.get_process_group_ranks(group))
        except Exception:
            pass
    return list(range(world_size))


def _tensor_device(torch, output):
    values = output if isinstance(output, (tuple, list)) else (output,)
    for value in values:
        if torch.is_tensor(value):
            return value.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _as_1d_long_tensor(torch, values, device):
    if values is None:
        return None
    if torch.is_tensor(values):
        tensor = values.detach()
    else:
        tensor = torch.as_tensor(values)
    if tensor.dim() == 0:
        tensor = tensor.reshape(1)
    tensor = tensor.reshape(-1).to(dtype=torch.long)
    if tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def _send_splits_from_routing_map(torch, token_dispatcher, device):
    routing_map = getattr(token_dispatcher, "routing_map", None)
    if routing_map is None:
        return None

    local_tokens_per_expert = routing_map.sum(dim=0).long()
    ep_size = getattr(token_dispatcher, "ep_size", None)
    if ep_size is None or ep_size <= 0:
        return None
    if local_tokens_per_expert.numel() % ep_size != 0:
        return None

    experts_per_ep = local_tokens_per_expert.numel() // ep_size
    send_splits = local_tokens_per_expert.reshape(ep_size, experts_per_ep).sum(dim=1)
    return _as_1d_long_tensor(torch, send_splits, device)


def _sync_dispatcher_metadata(token_dispatcher):
    d2h_event = getattr(token_dispatcher, "d2h_event", None)
    if d2h_event is not None:
        d2h_event.synchronize()


def _send_splits_from_dispatcher(torch, token_dispatcher, device):
    _sync_dispatcher_metadata(token_dispatcher)

    input_splits = getattr(token_dispatcher, "input_splits", None)
    if input_splits is not None:
        return _as_1d_long_tensor(torch, input_splits, device)

    return _send_splits_from_routing_map(torch, token_dispatcher, device)


def _format_moe_token_event(
    *,
    event_id,
    iteration,
    microbatch,
    phase,
    event,
    layer,
    rank,
    received_tokens,
):
    return (
        "moe_token_event: event_id={}, step={}, iteration={}, microbatch={}, "
        "phase={}, event={}, layer={}, rank={}, received_tokens={}".format(
            _format_field(event_id),
            _format_field(iteration),
            _format_field(iteration),
            _format_field(microbatch),
            _format_field(phase),
            _format_field(event),
            _format_field(layer),
            _format_field(rank),
            _format_field(received_tokens),
        )
    )


def _write_moe_token_event(
    *,
    event_id,
    iteration,
    microbatch,
    phase,
    event,
    layer,
    rank,
    received_tokens,
):
    print(
        _format_moe_token_event(
            event_id=event_id,
            iteration=iteration,
            microbatch=microbatch,
            phase=phase,
            event=event,
            layer=layer,
            rank=rank,
            received_tokens=received_tokens,
        ),
        flush=True,
    )


def _first_tensor(torch, output):
    values = output if isinstance(output, (tuple, list)) else (output,)
    for value in values:
        if torch.is_tensor(value):
            return value
    return None


def _observe_activation_demand(layer, output, args):
    import torch

    dispatched_tokens = _first_tensor(torch, output)
    if dispatched_tokens is None or not getattr(dispatched_tokens, "shape", None):
        return None

    from bgl2.core.models.gpt import gpt_activation
    from bgl2.core.pipeline_parallel.offload import (
        add_batch_to_offload_manager,
    )

    try:
        model = gpt_activation.get_activation_model()
    except RuntimeError:
        model = gpt_activation.initialize_activation_model(
            args=args,
            transformer_config=getattr(layer, "config", None),
        )

    model.set_dynamic_value("ntoken", int(dispatched_tokens.shape[0]))
    memory_bytes = gpt_activation.calculate_memory_consumption(args)
    return add_batch_to_offload_manager(memory_bytes)


def _register_free_event_hook(
    torch,
    output,
    *,
    event_id,
    iteration,
    microbatch,
    layer,
    rank,
    received_tokens,
):
    tensor = _first_tensor(torch, output)
    if tensor is None or not getattr(tensor, "requires_grad", False):
        return

    def write_free_event(grad):
        _write_moe_token_event(
            event_id=event_id,
            iteration=iteration,
            microbatch=microbatch,
            phase="backward",
            event="free",
            layer=layer,
            rank=rank,
            received_tokens=received_tokens,
        )
        return grad

    try:
        tensor.register_hook(write_free_event)
    except Exception:
        pass


def _capture_pre_dispatch_splits(layer, dispatch_args, dispatch_kwargs):
    import torch

    token_dispatcher = getattr(layer, "token_dispatcher", None)
    if token_dispatcher is None:
        return None

    reference_values = dispatch_args
    if not reference_values:
        reference_values = tuple(dispatch_kwargs.values())
    device = _tensor_device(torch, reference_values)
    return _send_splits_from_dispatcher(torch, token_dispatcher, device)


def _write_imbalance_metrics(args, layer, recv_counts, step):
    layer_idx = getattr(layer, "layer_number", "unknown")
    token_counts = [received_tokens for _, received_tokens in recv_counts]
    average_tokens = sum(token_counts) / len(token_counts)
    max_tokens = max(token_counts)
    imbalance = max_tokens / average_tokens if average_tokens > 0 else float("inf")

    for receiver_rank, received_tokens in recv_counts:
        print(
            "step: {}, layer: {}, rank: {}, received_tokens: {}".format(
                step, layer_idx, receiver_rank, received_tokens
            ),
            flush=True,
        )

    print(
        "step: {}, layer: {}, average_received_tokens: {}, "
        "max_received_tokens: {}, imbalance_ratio: {:.2f}".format(
            step, layer_idx, average_tokens, max_tokens, imbalance
        ),
        flush=True,
    )

    from megatron.training.global_vars import get_tensorboard_writer

    writer = get_tensorboard_writer()
    if writer:
        writer.add_scalar(f"moe/imbalance_layer_{layer_idx}", imbalance, step)
        writer.add_scalar(f"moe/max_tokens_layer_{layer_idx}", max_tokens, step)
        writer.flush()


def _export_moe_imbalance_ratio(layer, output, args, local_splits=None):
    import torch

    if not _is_dist_initialized(torch):
        return

    token_dispatcher = getattr(layer, "token_dispatcher", None)
    if token_dispatcher is None:
        return

    group = getattr(token_dispatcher, "ep_group", None)
    if group is None:
        return

    world_size = _group_size(torch, group)
    if world_size <= 0:
        return

    if local_splits is None:
        device = _tensor_device(torch, output)
        local_splits = _send_splits_from_dispatcher(torch, token_dispatcher, device)
    if local_splits is None:
        return
    if local_splits.numel() != world_size:
        rank = torch.distributed.get_rank()
        print(
            "[Rank {}] MoE imbalance stats skipped: split count {} != EP world size {}".format(
                rank, local_splits.numel(), world_size
            ),
            flush=True,
        )
        return

    if world_size == 1:
        global_matrix = local_splits.reshape(1, -1)
    else:
        gathered_splits = [torch.zeros_like(local_splits) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_splits, local_splits, group=group)
        global_matrix = torch.stack(gathered_splits)

    recv_token_counts = [int(value) for value in global_matrix.sum(dim=0).tolist()]
    group_ranks = _group_ranks(torch, group, world_size)
    local_group_rank = _group_rank(torch, group)
    if local_group_rank < 0 or local_group_rank >= len(group_ranks):
        return

    iteration = _current_iteration(args)
    step = _step_for_metrics(iteration)
    microbatch = _current_microbatch(layer, args)
    event_id = _next_event_id()
    layer_idx = getattr(layer, "layer_number", UNKNOWN_FIELD)
    local_rank = group_ranks[local_group_rank]
    local_received_tokens = recv_token_counts[local_group_rank]

    _write_moe_token_event(
        event_id=event_id,
        iteration=iteration,
        microbatch=microbatch,
        phase="forward",
        event="alloc",
        layer=layer_idx,
        rank=local_rank,
        received_tokens=local_received_tokens,
    )
    _register_free_event_hook(
        torch,
        output,
        event_id=event_id,
        iteration=iteration,
        microbatch=microbatch,
        layer=layer_idx,
        rank=local_rank,
        received_tokens=local_received_tokens,
    )

    recv_counts = list(zip(group_ranks, recv_token_counts))

    if local_group_rank == world_size - 1:
        _write_imbalance_metrics(args, layer, recv_counts, step)


def moe_dispatch_wrapper(dispatch):
    """Wrap MoELayer.dispatch to export token lifecycle and imbalance metrics."""

    @wraps(dispatch)
    def wrapper(self, *args, **kwargs):
        runtime_args = _get_megatron_args()
        megatron_args = _get_enabled_args(runtime_args)
        local_splits = None
        if megatron_args is not None:
            try:
                local_splits = _capture_pre_dispatch_splits(self, args, kwargs)
            except Exception:
                local_splits = None

        output = dispatch(self, *args, **kwargs)
        if _dynamic_offload_enabled(runtime_args):
            _observe_activation_demand(self, output, runtime_args)
        if megatron_args is None:
            return output

        try:
            _export_moe_imbalance_ratio(
                self,
                output,
                megatron_args,
                local_splits=local_splits,
            )
        except Exception as exc:
            print(
                "[Rank {}] Failed to export MoE token dispatch stats: {}".format(
                    _rank_for_log(), exc
                ),
                flush=True,
            )
        return output

    return wrapper


def moe_router_and_preprocess_wrapper(router_and_preprocess):
    """Wrap MoELayer.router_and_preprocess to export dispatch imbalance metrics."""

    @wraps(router_and_preprocess)
    def wrapper(self, hidden_states):
        output = router_and_preprocess(self, hidden_states)
        args = _get_enabled_args()
        if args is None:
            return output

        try:
            _export_moe_imbalance_ratio(self, output, args)
        except Exception as exc:
            print(
                "[Rank {}] Failed to export MoE imbalance stats: {}".format(
                    _rank_for_log(), exc
                ),
                flush=True,
            )
        return output

    return wrapper


def transformer_layer_forward_mlp_wrapper(forward_mlp):
    """Propagate TransformerLayer microbatch ids to nested MLP/MoE modules."""

    @wraps(forward_mlp)
    def wrapper(self, *args, **kwargs):
        mlp = getattr(self, "mlp", None)
        microbatch = getattr(self, "current_microbatch", None)
        if mlp is not None and microbatch is not None:
            try:
                mlp.current_microbatch = microbatch
            except Exception:
                pass
        from bgl2.core.pipeline_parallel.offload import mlp_activation_offload_scope

        layer_number = getattr(self, "layer_number", getattr(mlp, "layer_number", "unknown"))
        with mlp_activation_offload_scope(layer_number):
            return forward_mlp(self, *args, **kwargs)

    return wrapper
