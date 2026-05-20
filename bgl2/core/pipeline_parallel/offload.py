"""Activation offload runtime aligned with the dcu_megatron implementation.

The offload selection model is intentionally dcu-style: patched TE/fused ops call
``pack_hook(..., op_name=...)`` and only tensors whose ``op_name`` is listed in
``offload_modules``/``bgl2_offload_modules`` are recorded for offload. The
pipeline runtime at the bottom is a thin compatibility adapter for bgl2's
current scheduler context calls.
"""

import contextlib
import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass

import torch

from megatron.training import get_args

try:
    from deep_gemm.jit_kernels.quant import fp8_dequant_per_tile, fp8_quant_per_tile

    has_deep_gemm = True
except Exception:
    has_deep_gemm = False


def _get_arg(name, default=None):
    return getattr(get_args(), name, default)


def _any_arg_true(*names):
    return any(bool(_get_arg(name, False)) for name in names)


def _patch_te_enabled():
    return _any_arg_true(
        "bgl2_patch_te",
        "bgl2_enable_activation_offload",
    )


def _offload_modules():
    modules = _get_arg("offload_modules", None)
    if modules is None:
        modules = _get_arg("bgl2_offload_modules", None)
    if modules is None:
        return set()
    if isinstance(modules, str):
        return {modules}
    return set(modules)


def _offload_min_bytes():
    return int(_get_arg("offload_min_bytes", _get_arg("bgl2_offload_min_bytes", 1024 * 1024)))


def _activation_offload_ratio():
    ratio = _get_arg("activation_offload_ratio", _get_arg("bgl2_activation_offload_ratio", 1.0))
    if not isinstance(ratio, (float, int)):
        from megatron.core import parallel_state

        ratio = ratio[parallel_state.get_pipeline_model_parallel_rank()]
    return float(ratio)


def _activation_offload_threshold():
    return _get_arg(
        "activation_offload_threshold",
        _get_arg("bgl2_activation_offload_threshold", None),
    )


def _per_batch_offload_size_mib():
    return _get_arg("per_batch_offload_size", _get_arg("bgl2_per_batch_offload_size", 0))


def _activation_offload_stages():
    return max(1, int(_get_arg("activation_offload_stages", _get_arg("bgl2_activation_offload_stages", 1))))


def _prefetch_activation_reload():
    return bool(
        _get_arg(
            "prefetch_activation_reload",
            _get_arg("bgl2_prefetch_activation_reload", False),
        )
    )


def _offload_granularity():
    return _get_arg("bgl2_offload_granularity", "microbatch")


def _layer_offload_enabled():
    return (
        bool(_get_arg("bgl2_enable_activation_offload", False))
        and _offload_granularity() == "layer"
    )


def _configured_offload_limit_bytes():
    size_mib = _per_batch_offload_size_mib()
    if size_mib is None or size_mib <= 0:
        return 0
    return int(size_mib * 2**20)


def _debug_enabled():
    return os.getenv("BGL2_OFFLOAD_DEBUG", "").lower() in {"1", "true", "yes", "on"}


def _debug_rank_allowed(rank):
    ranks = os.getenv("BGL2_OFFLOAD_DEBUG_RANKS")
    if not ranks:
        return True
    allowed = {item.strip() for item in ranks.replace(",", " ").split() if item.strip()}
    return str(rank) in allowed


def _debug_rank_info():
    rank = os.getenv("RANK", "?")
    local_rank = os.getenv("LOCAL_RANK", "?")
    pp_rank = "?"

    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
    except Exception:
        pass

    try:
        from megatron.core import parallel_state

        if parallel_state.is_initialized():
            pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    except Exception:
        pass

    return rank, local_rank, pp_rank


def debug_log(event, **fields):
    if not _debug_enabled():
        return

    rank, local_rank, pp_rank = _debug_rank_info()
    if not _debug_rank_allowed(rank):
        return

    details = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
    print(
        f"[BGL2-OFFLOAD-DEBUG] rank={rank} local_rank={local_rank} pp_rank={pp_rank} "
        f"event={event} {details}",
        flush=True,
    )


def _debug_cuda_memory_fields():
    if not _debug_enabled():
        return {}
    try:
        is_available = getattr(torch.cuda, "is_available", None)
        if callable(is_available) and not is_available():
            return {}
        return {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
    except Exception:
        return {}


def debug_memory_log(event, **fields):
    if not _debug_enabled():
        return
    fields.update(_debug_cuda_memory_fields())
    fields["timestamp_ns"] = time.time_ns()
    debug_log(event, **fields)


def _offload_size_bytes(total_bytes, group_num):
    mib = 2**20
    per_batch_offload_size = _per_batch_offload_size_mib()

    if per_batch_offload_size is None or per_batch_offload_size <= 0:
        return 0

    offload_size = (
        total_bytes
        if total_bytes < per_batch_offload_size * mib
        else int(per_batch_offload_size * mib)
    )
    if offload_size % group_num != 0:
        raise AssertionError(
            "MB is not divisible by the offload group num! Please switch to a power of 2."
        )
    return offload_size


def _make_key(virtual_microbatch_id, model_chunk_id):
    return model_chunk_id, virtual_microbatch_id


first_reporter = True


class MemoryReporter:
    def __init__(self, print_ranks=(), enabled=True):
        global first_reporter
        self.enabled = enabled
        self.base_mem = 0
        self.max_mem = 0
        if not enabled:
            self.rank = -1
            self.print = False
            return
        self.rank = torch.distributed.get_rank()
        self.print = self.rank in print_ranks and first_reporter
        first_reporter = False

    def reset(self):
        if not self.enabled:
            return
        self.base_mem = torch.cuda.memory_allocated()
        self.max_mem = self.base_mem

    def record(self):
        if not self.enabled:
            return
        self.max_mem = max(self.max_mem, torch.cuda.memory_allocated())

    def report(self):
        if not self.enabled or not self.print:
            return

        from bgl2.core.models.gpt import gpt_activation

        model = gpt_activation.get_activation_model()
        offload_modules = _offload_modules()
        print(
            f"MemoryReporter[{self.rank}]: Expected non offload enabled memory "
            f"{model.calculate(except_modules=offload_modules)/1024**2:.2f} MB, "
            f"offload enabled memory "
            f"{model.calculate(selected_modules=offload_modules)/1024**2:.2f} MB, "
            f"offload ratio {get_offload_ratio()}"
        )
        print(
            f"MemoryReporter[{self.rank}]: {(self.max_mem - self.base_mem) / 1024 / 1024:.2f} MB used, "
            f"{self.base_mem / 1024 / 1024:.2f} MB base"
        )


class OffloadManager:
    def __init__(self, debug=False):
        self.max_active_batch = 0
        self.num_active_batch = 0
        self.offload_ratio = 0.0
        self.memory_consumption = 0
        self.offload_sizes = {}
        self.debug = debug
        self.memory_limit = None

    def reset(self, num_warmup_batches, memory_limit_MB):
        self.num_active_batch = 0
        self.max_active_batch = num_warmup_batches + 3
        if memory_limit_MB is not None:
            self.memory_limit = memory_limit_MB * 1024 * 1024
        else:
            self.memory_limit = None
        self.offload_ratio = 0.0
        self.memory_consumption = 0
        self.offload_sizes = {}

    def add_batch(self, memory_consumption, batch_id):
        if self.memory_limit is None:
            if self.debug:
                my_rank = torch.distributed.get_rank()
                print(f"OffloadManager[{my_rank}]: no memory limit, skipping")
            return

        offload_memory = max(
            0,
            self.memory_consumption + memory_consumption - self.memory_limit,
        )

        if self.num_active_batch / self.max_active_batch > 0.1:
            active_batch = self.num_active_batch + 1
            memory = self.memory_consumption + memory_consumption
            avg_memory = memory / active_batch
            to_offload = (
                avg_memory * (self.max_active_batch - 1) - self.memory_limit
            ) / (self.max_active_batch - active_batch)
            offload_memory = max(offload_memory, to_offload)

        offload_memory = min(offload_memory, memory_consumption)
        self.offload_ratio = offload_memory / memory_consumption
        self.memory_consumption += memory_consumption - offload_memory
        self.num_active_batch += 1
        self.offload_sizes[batch_id] = memory_consumption - offload_memory
        self.print_debug(memory_consumption)

    def get_offload_ratio(self):
        if self.memory_limit is None:
            assert False
        return self.offload_ratio

    def remove_batch(self, batch_id):
        if self.memory_limit is None:
            return
        resident_memory = self.offload_sizes.pop(batch_id, None)
        if resident_memory is None:
            return
        self.memory_consumption -= resident_memory
        self.num_active_batch -= 1

    def print_debug(self, new_mem):
        if not self.debug:
            return

        my_rank = torch.distributed.get_rank()
        print(
            f"OffloadManager[{my_rank}]: {self.num_active_batch} active batches, "
            f"{self.memory_consumption / 1024 / 1024:.2f} MB used, "
            f"{self.offload_ratio:.2f} offload ratio {new_mem / 1024 / 1024:.2f} MB new batch, "
            f"{self.memory_limit / 1024 / 1024:.2f} MB limit"
        )


_OFFLOAD_MANAGER = OffloadManager()


def reset_offload_manager(num_warmup_batches, memory_limit_MB=None):
    if memory_limit_MB is None:
        memory_limit_MB = _activation_offload_threshold()
    _OFFLOAD_MANAGER.reset(num_warmup_batches, memory_limit_MB)


_F_BATCH_ID = None


def set_current_forward_batch_id(batch_id):
    global _F_BATCH_ID
    _F_BATCH_ID = batch_id


def clear_current_forward_batch_id():
    global _F_BATCH_ID
    _F_BATCH_ID = None


_B_BATCH_ID = None


def set_current_backward_batch_id(batch_id):
    global _B_BATCH_ID
    _B_BATCH_ID = batch_id


def clear_current_backward_batch_id():
    global _B_BATCH_ID
    _B_BATCH_ID = None


def add_batch_to_offload_manager(memory_consumption, batch_id=None):
    global _F_BATCH_ID
    if batch_id is None:
        batch_id = _F_BATCH_ID
        _F_BATCH_ID = None
    if batch_id is None:
        return None
    _OFFLOAD_MANAGER.add_batch(memory_consumption, batch_id)


def remove_batch_from_offload_manager(batch_id=None):
    global _B_BATCH_ID
    if batch_id is None:
        batch_id = _B_BATCH_ID
        _B_BATCH_ID = None
    _OFFLOAD_MANAGER.remove_batch(batch_id)


def get_offload_ratio():
    if _activation_offload_threshold() is not None:
        return _OFFLOAD_MANAGER.get_offload_ratio()
    return _activation_offload_ratio()


_MEMCPY_STREAM = {}
_GPU_BUFFER_POOL = {}
_PINNED_BUFFER_POOL = defaultdict(list)
_PINNED_BUFFER_POOL_STATS = defaultdict(int)
# Small buckets plus modest growth headroom amortize changing MoE shapes
# without the memory overhead of power-of-two slabs.
_PINNED_BUFFER_GRANULARITY_BYTES = 64 * 2**20
_PINNED_BUFFER_GROWTH_NUMERATOR = 9
_PINNED_BUFFER_GROWTH_DENOMINATOR = 8


def get_memcpy_stream(key):
    if key not in ("onload", "offload"):
        raise AssertionError("unsupported stream")
    if key not in _MEMCPY_STREAM:
        _MEMCPY_STREAM[key] = torch.cuda.Stream()
    return _MEMCPY_STREAM[key]


def get_persistent_gpu_buffer(key, size):
    previous_capacity = _GPU_BUFFER_POOL[key].numel() if key in _GPU_BUFFER_POOL else 0
    if previous_capacity < size:
        before = _debug_cuda_memory_fields()
        _GPU_BUFFER_POOL[key] = None
        _GPU_BUFFER_POOL[key] = torch.empty(size, dtype=torch.uint8, device="cuda")
        _GPU_BUFFER_POOL[key].ref_cnt = 0
        if key.startswith("onload"):
            debug_memory_log(
                "reload-malloc",
                buffer_key=key,
                requested_bytes=size,
                previous_capacity_bytes=previous_capacity,
                cuda_allocated_before_bytes=before.get("cuda_allocated_bytes"),
            )
    return _GPU_BUFFER_POOL[key][:size]


def _release_persistent_gpu_buffer(buffer):
    if buffer is None or getattr(buffer, "ref_cnt", 0) != 0:
        return
    for key, pooled_buffer in tuple(_GPU_BUFFER_POOL.items()):
        if pooled_buffer is buffer:
            _GPU_BUFFER_POOL.pop(key)


def _pop_pinned_buffer(key):
    buffers = _PINNED_BUFFER_POOL[key]
    buffer = buffers.pop()
    if not buffers:
        del _PINNED_BUFFER_POOL[key]
    return buffer


def _pinned_buffer_allocation_capacity(num_elements, dtype):
    itemsize = getattr(dtype, "itemsize", 1)
    if callable(itemsize):
        itemsize = itemsize()
    granularity = max(1, _PINNED_BUFFER_GRANULARITY_BYTES // int(itemsize))
    grown_elements = (
        num_elements * _PINNED_BUFFER_GROWTH_NUMERATOR
        + _PINNED_BUFFER_GROWTH_DENOMINATOR
        - 1
    ) // _PINNED_BUFFER_GROWTH_DENOMINATOR
    return ((grown_elements + granularity - 1) // granularity) * granularity


def get_cpu_buffer(num_bytes, dtype=torch.uint8):
    candidate_key = min(
        (
            key
            for key, buffers in _PINNED_BUFFER_POOL.items()
            if buffers and key[1] == dtype and key[0] >= num_bytes
        ),
        key=lambda key: key[0],
        default=None,
    )
    if candidate_key is None:
        # Replace the least useful free slab when demand grows. This lets the
        # pool track peak concurrency instead of retaining every historical shape.
        replacement_key = min(
            (
                key
                for key, buffers in _PINNED_BUFFER_POOL.items()
                if buffers and key[1] == dtype
            ),
            key=lambda key: key[0],
            default=None,
        )
        replaced_capacity = 0
        if replacement_key is not None:
            replaced_buffer = _pop_pinned_buffer(replacement_key)
            replaced_capacity = replaced_buffer.numel()
            element_size = getattr(replaced_buffer, "element_size", None)
            replaced_bytes = replaced_capacity * (
                element_size() if callable(element_size) else 1
            )
            _PINNED_BUFFER_POOL_STATS["replacements"] += 1
            _PINNED_BUFFER_POOL_STATS["retired_bytes"] += replaced_bytes
            del replaced_buffer

        pin_memory = bool(_get_arg("bgl2_offload_pin_memory", True))
        capacity = _pinned_buffer_allocation_capacity(num_bytes, dtype)
        buffer = torch.empty(capacity, dtype=dtype, pin_memory=pin_memory)
        element_size = getattr(buffer, "element_size", None)
        allocated_bytes = buffer.numel() * (element_size() if callable(element_size) else 1)
        _PINNED_BUFFER_POOL_STATS["misses"] += 1
        _PINNED_BUFFER_POOL_STATS["allocated_bytes"] += allocated_bytes
        debug_log(
            "pinned-buffer-allocate",
            requested_elements=num_bytes,
            capacity_elements=buffer.numel(),
            allocated_bytes=allocated_bytes,
            cumulative_allocated_bytes=_PINNED_BUFFER_POOL_STATS["allocated_bytes"],
            cumulative_retired_bytes=_PINNED_BUFFER_POOL_STATS["retired_bytes"],
            hits=_PINNED_BUFFER_POOL_STATS["hits"],
            managed_capacity_bytes=(
                _PINNED_BUFFER_POOL_STATS["allocated_bytes"]
                - _PINNED_BUFFER_POOL_STATS["retired_bytes"]
            ),
            misses=_PINNED_BUFFER_POOL_STATS["misses"],
            replaced_capacity_elements=replaced_capacity,
            replacements=_PINNED_BUFFER_POOL_STATS["replacements"],
        )
    else:
        buffer = _pop_pinned_buffer(candidate_key)
        _PINNED_BUFFER_POOL_STATS["hits"] += 1
        debug_log(
            "pinned-buffer-reuse",
            requested_elements=num_bytes,
            capacity_elements=buffer.numel(),
            hits=_PINNED_BUFFER_POOL_STATS["hits"],
            misses=_PINNED_BUFFER_POOL_STATS["misses"],
        )
    return buffer[:num_bytes]


def recycle_cpu_buffer(buffer):
    pooled_buffer = buffer
    while getattr(pooled_buffer, "_base", None) is not None:
        pooled_buffer = pooled_buffer._base
    _PINNED_BUFFER_POOL[pooled_buffer.numel(), pooled_buffer.dtype].append(pooled_buffer)
    if _debug_enabled():
        debug_log(
            "pinned-buffer-recycle",
            capacity_elements=pooled_buffer.numel(),
            pooled_buffers=sum(len(buffers) for buffers in _PINNED_BUFFER_POOL.values()),
        )


def copy2d_(dst, src):
    assert dst.dtype == src.dtype, "dtype mismatch"
    if not dst.is_contiguous():
        raise NotImplementedError(f"unsupported dst shape {dst.shape} stride {dst.stride()}")
    shape = src.shape
    stride = src.stride()
    if stride[-1] == 1 and all(
        stride[i] == shape[i + 1] * stride[i + 1] for i in range(0, len(shape) - 2)
    ):
        try:
            import wrap_gemm_cuda

            dw = src.dtype.itemsize
            cudaMemcpyDefault = 4
            wrap_gemm_cuda.wrap_cuda_memcpy_2d_async(
                dst.data_ptr(),
                shape[-1] * dw,
                src.data_ptr(),
                stride[-2] * dw,
                shape[-1] * dw,
                shape[:-1].numel(),
                cudaMemcpyDefault,
                torch.cuda.current_stream().cuda_stream,
            )
        except ImportError:
            dst.copy_(src.contiguous())
    else:
        raise NotImplementedError(f"unsupported src shape {shape} stride {stride}")


def fast_contiguous(x):
    if x.is_contiguous():
        return x
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    copy2d_(out, x)
    return out


def _unwrap_tensor_meta(x):
    if isinstance(x, torch.Tensor):
        return x, x.shape, x.dtype, x.device

    candidate_attrs = ("_data", "data", "_tensor", "tensor")
    for attr in candidate_attrs:
        if hasattr(x, attr):
            inner = getattr(x, attr)
            if isinstance(inner, torch.Tensor):
                return inner, inner.shape, inner.dtype, inner.device

    if all(hasattr(x, key) for key in ("shape", "dtype", "device")):
        return x, x.shape, x.dtype, x.device

    raise TypeError(f"Unsupported tensor-like object: {type(x)}")


class TensorWrap:
    def __init__(self, x, scales=None):
        inner, shape, dtype, device = _unwrap_tensor_meta(x)
        self.x = inner
        self.outer = x
        self.scales = scales
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.base = None
        self.activation_group = None
        self.release_after_unpack = False


class TensorPack:
    def __init__(self, tensor_wrap, op_name=None, **kwargs):
        self.tensor_wrap = tensor_wrap
        self.op_name = op_name
        self.extra_kwargs = kwargs

    def get(self):
        if self.tensor_wrap.x is None and self.tensor_wrap.activation_group is not None:
            self.tensor_wrap.activation_group.ensure_onloaded()
        return self.tensor_wrap.x

    def get_scales(self):
        return self.tensor_wrap.scales

    def release(self):
        tensor_wrap = getattr(self, "tensor_wrap", None)
        if tensor_wrap is None:
            return
        tensor_wrap.x = None
        tensor_wrap.outer = None
        tensor_wrap.scales = None
        tensor_wrap.activation_group = None
        base = tensor_wrap.base
        tensor_wrap.base = None
        if base is not None:
            base.ref_cnt -= 1
            _release_persistent_gpu_buffer(base)

    def __del__(self):
        self.release()


class CopyTaskGroup:
    def __init__(self, total_offload_size, offload_group_num):
        self.total_offload_size = total_offload_size
        self.offload_group_num = offload_group_num
        self.group_size = total_offload_size // offload_group_num
        self.current_copy_task_id = 0
        self.current_copy_task_size = 0
        self.copy_task_groups = [[] for _ in range(offload_group_num)]
        self.print = torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() in []

    def add_tensor(self, begin_idx, end_idx, tensor):
        assert end_idx - begin_idx == tensor.numel(), (
            f"{begin_idx=} {end_idx=} {tensor.shape=} {tensor.dtype=}"
        )
        assert tensor.is_contiguous(), f"tensor {tensor.shape} is not contiguous"
        tensor = tensor.view(-1)

        memory_left_to_process = end_idx - begin_idx
        while memory_left_to_process != 0:
            if self.current_copy_task_size + memory_left_to_process <= self.group_size:
                self.copy_task_groups[self.current_copy_task_id].append((begin_idx, end_idx, tensor))
                self.current_copy_task_size += memory_left_to_process
                break

            memory_to_process = self.group_size - self.current_copy_task_size
            if memory_to_process != 0:
                self.copy_task_groups[self.current_copy_task_id].append(
                    (begin_idx, begin_idx + memory_to_process, tensor[:memory_to_process])
                )
                self.current_copy_task_size += memory_to_process
                begin_idx += memory_to_process
                memory_left_to_process -= memory_to_process
                tensor = tensor[memory_to_process:]

            self.current_copy_task_id += 1
            self.current_copy_task_size = 0

    def get_task_groups(self):
        return self.copy_task_groups


print_once = False


class ActivationGroup:
    def __init__(
        self,
        tensors,
        key,
        offload_groups,
        *,
        offload_limit_bytes=None,
        release_source_storage=True,
    ):
        self.key = key
        live_tensors = (tensor for tensor in tensors if tensor.x is not None)
        self.tensors = sorted(
            live_tensors,
            key=lambda tensor: (not tensor.x.is_contiguous(), -tensor.shape.numel()),
        )
        self.offload_ratio = get_offload_ratio()
        self.offload_group_num = max(1, offload_groups)
        self.offload_limit_bytes = offload_limit_bytes
        self.release_source_storage = release_source_storage
        self.offload_groups = None
        self.remained_not_offloaded = None
        self._onloaded = False
        if self.offload_ratio > 0.5:
            self.tensors = self.tensors[::-1]

    def offload_prologue(self, *, use_bucket):
        if not self.tensors:
            return None, None, None
        self.map = []
        top = 0
        for i, tensor in enumerate(self.tensors):
            duplicate_flag = False
            if tensor.x.is_contiguous():
                for j, prev_tensor in enumerate(self.tensors[:i]):
                    if (
                        tensor.x.data_ptr() == prev_tensor.x.data_ptr()
                        and prev_tensor.x.is_contiguous()
                        and tensor.device == prev_tensor.device
                        and tensor.shape.numel() == prev_tensor.shape.numel()
                    ):
                        begin_idx, end_idx, _ = self.map[j]
                        duplicate_flag = True
                        self.map.append((begin_idx, end_idx, duplicate_flag))
                        break
            if not duplicate_flag:
                nbytes = tensor.shape.numel() * tensor.dtype.itemsize
                self.map.append((top, top + nbytes, duplicate_flag))
                top += nbytes

        mib = 2**20
        per_batch_offload_size = _per_batch_offload_size_mib()
        global print_once
        if not print_once:
            print(f"{top/mib=} {per_batch_offload_size=}")
            print_once = True
        offload_size = _offload_size_bytes(top, self.offload_group_num)
        if self.offload_limit_bytes is not None:
            offload_size = min(offload_size, self.offload_limit_bytes)
            offload_size -= offload_size % self.offload_group_num
        debug_log(
            "activation_prologue",
            key=self.key,
            tensors=len(self.tensors),
            total_mib=top / mib,
            per_batch_offload_size=per_batch_offload_size,
            offload_size=offload_size,
            offload_limit_bytes=self.offload_limit_bytes,
            group_num=self.offload_group_num,
        )

        if offload_size == 0:
            self.offload_groups = []
            self.buffer_cpu = None
            self.use_bucket = use_bucket
            debug_log("activation_zero_noop", key=self.key)
            return None, None, None

        if use_bucket:
            buffer = get_persistent_gpu_buffer("offload", offload_size)
        else:
            buffer = None

        assert offload_size % self.offload_group_num == 0, (
            "MB is not divisible by the offload group num! Please switch to a power of 2."
        )
        groups = CopyTaskGroup(offload_size, self.offload_group_num)
        self.group_size = offload_size // self.offload_group_num

        partially_offloaded_bases = set()
        for tensor, (begin_idx, end_idx, duplicate_flag) in zip(self.tensors, self.map):
            assert tensor.x.device.type == "cuda"
            if end_idx <= offload_size:
                if not duplicate_flag:
                    if tensor.x._base is not None:
                        partially_offloaded_bases.add(tensor.x._base)
                    if use_bucket:
                        buffer[begin_idx:end_idx].view(tensor.dtype).view(tensor.shape).copy_(tensor.x)
                    else:
                        groups.add_tensor(begin_idx, end_idx, tensor.x.view(torch.uint8))
                tensor.x = None
            elif begin_idx < offload_size:
                if not duplicate_flag:
                    if tensor.x._base is not None:
                        partially_offloaded_bases.add(tensor.x._base)
                    linear_data = fast_contiguous(tensor.x).view(-1).view(torch.uint8)
                    if use_bucket:
                        buffer[begin_idx:].copy_(linear_data[: offload_size - begin_idx])
                    else:
                        groups.add_tensor(begin_idx, offload_size, linear_data[: offload_size - begin_idx])
                    self.remained_not_offloaded = linear_data[offload_size - begin_idx :].clone()
                tensor.x = None
            elif tensor.x._base in partially_offloaded_bases:
                if duplicate_flag:
                    raise NotImplementedError("does not support partially offload duplicate tensors")
                tensor.x = tensor.x.clone()

        self.offload_groups = groups.get_task_groups()
        self.buffer_cpu = get_cpu_buffer(offload_size)
        self.use_bucket = use_bucket

    def offload_now(self):
        self.offload_prologue(use_bucket=False)
        if not self.offload_groups or self.buffer_cpu is None:
            return 0
        for tensor in self.tensors:
            tensor.activation_group = self
            tensor.release_after_unpack = True
        for group_id in range(self.offload_group_num):
            self.offload_issue(group_id)
        offloaded_bytes = self.buffer_cpu.numel()
        self.offload_epilogue()
        return offloaded_bytes

    def offload_issue(self, group_id):
        if not self.offload_groups:
            return
        debug_log("offload_issue_begin", key=self.key, group_id=group_id)
        copy_tasks = self.offload_groups[group_id]
        stream = get_memcpy_stream("offload")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            if self.use_bucket:
                raise AssertionError("unsupported")
            for begin_idx, end_idx, x in copy_tasks:
                if x.is_contiguous():
                    msg = f"{begin_idx=} {end_idx=} {x.shape=} {x.dtype=}"
                    assert end_idx - begin_idx == x.numel(), msg
                    self.buffer_cpu[begin_idx * x.element_size() : end_idx * x.element_size()].view(
                        x.dtype
                    ).view(x.shape).copy_(x, non_blocking=True)
                else:
                    copy2d_(self.buffer_cpu[begin_idx:end_idx].view(x.dtype).view(x.shape), x)
                record_stream = getattr(x, "record_stream", None)
                if callable(record_stream):
                    record_stream(stream)
        debug_log("offload_issue_end", key=self.key, group_id=group_id, tasks=len(copy_tasks))

    def offload_epilogue(self):
        if not self.tensors or not self.offload_groups:
            return
        debug_log("offload_epilogue_begin", key=self.key)
        stream = get_memcpy_stream("offload")
        torch.cuda.current_stream().wait_stream(stream)
        offloaded_bytes = self.buffer_cpu.numel()
        before_free = _debug_cuda_memory_fields()
        debug_memory_log(
            "offload",
            key=self.key,
            bytes=offloaded_bytes,
            tensors=len(self.tensors),
            groups=self.offload_group_num,
        )
        if self.release_source_storage:
            for copy_tasks in self.offload_groups:
                for _, _, x in copy_tasks:
                    x.untyped_storage().resize_(0)
        else:
            # A layer source may alias a tensor that is still live in forward.
            # Drop only bgl2's references and let PyTorch preserve real users.
            for tensor in self.tensors:
                if tensor.x is None:
                    tensor.outer = None
            self.offload_groups = [[] for _ in range(self.offload_group_num)]
        after_free = _debug_cuda_memory_fields()
        before_allocated = before_free.get("cuda_allocated_bytes")
        after_allocated = after_free.get("cuda_allocated_bytes")
        freed_bytes = None
        if before_allocated is not None and after_allocated is not None:
            freed_bytes = max(0, before_allocated - after_allocated)
        debug_memory_log(
            "offload-free",
            key=self.key,
            bytes=offloaded_bytes,
            freed_bytes=freed_bytes,
            cuda_allocated_before_bytes=before_allocated,
        )
        debug_log("offload_epilogue_end", key=self.key)

    def onload_prologue(
        self,
        *,
        overlap_d2h_h2d,
        ping_pong_onload,
        onload_buffer_key=None,
    ):
        if not self.tensors or not self.offload_groups or self.buffer_cpu is None:
            return None, None, ping_pong_onload
        debug_log("onload_prologue_begin", key=self.key)
        stream_key = "onload" if overlap_d2h_h2d else "offload"
        if onload_buffer_key is not None:
            buffer_key = onload_buffer_key
        elif ping_pong_onload:
            buffer_key = "onload_ping"
            buffer = get_persistent_gpu_buffer(buffer_key, self.buffer_cpu.numel())
            if buffer._base.ref_cnt > 0:
                buffer_key = "onload_pong"
        else:
            buffer_key = stream_key
        stream = get_memcpy_stream(stream_key)
        buffer = get_persistent_gpu_buffer(buffer_key, self.buffer_cpu.numel())
        assert buffer._base.ref_cnt == 0, f"{self.key=} {buffer._base.ref_cnt=} {buffer_key=}"
        stream.wait_stream(torch.cuda.current_stream())
        self.onload_stream = stream
        self.onload_buffer = buffer
        self.onload_size = self.buffer_cpu.numel()
        self.onload_size_per_group = self.onload_size // self.offload_group_num
        assert self.onload_size % self.offload_group_num == 0, "onload size is not divisible by group num"
        debug_log("onload_prologue_end", key=self.key, onload_size=self.onload_size)
        return stream, buffer, ping_pong_onload

    def onload_issue(self, group_id):
        if not self.offload_groups or self.buffer_cpu is None:
            return
        debug_log("onload_issue_begin", key=self.key, group_id=group_id)
        stream = self.onload_stream
        buffer = self.onload_buffer[
            group_id * self.onload_size_per_group : (group_id + 1) * self.onload_size_per_group
        ]
        buffer_cpu = self.buffer_cpu[
            group_id * self.onload_size_per_group : (group_id + 1) * self.onload_size_per_group
        ]
        with torch.cuda.stream(stream):
            buffer.copy_(buffer_cpu, non_blocking=True)
        debug_log("onload_issue_end", key=self.key, group_id=group_id)

    def onload_epilogue(self, stream, buffer, ping_pong_onload):
        if not self.tensors or not self.offload_groups or self.buffer_cpu is None:
            return
        debug_log("onload_epilogue_begin", key=self.key)
        torch.cuda.current_stream().wait_stream(stream)
        debug_memory_log(
            "reload",
            key=self.key,
            bytes=buffer.numel(),
            tensors=len(self.tensors),
            groups=self.offload_group_num,
            ping_pong=ping_pong_onload,
        )
        recycle_cpu_buffer(self.buffer_cpu)
        self.buffer_cpu = None
        offload_size = buffer.numel()
        restored_tensors = {}
        for tensor, (begin_idx, end_idx, duplicate_flag) in zip(self.tensors, self.map):
            tensor_key = (begin_idx, end_idx)
            if end_idx <= offload_size:
                if duplicate_flag:
                    tensor.x = restored_tensors[tensor_key]
                else:
                    tensor.x = buffer[begin_idx:end_idx].view(tensor.dtype).view(tensor.shape)
                    if not ping_pong_onload:
                        tensor.x = tensor.x.clone()
                    restored_tensors[tensor_key] = tensor.x
                if ping_pong_onload:
                    tensor.base = buffer._base
                    tensor.base.ref_cnt += 1
            elif begin_idx < offload_size:
                if not duplicate_flag:
                    tensor.x = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
                    linear_data = tensor.x.view(-1).view(buffer.dtype)
                    restored_bytes = offload_size - begin_idx
                    linear_data[:restored_bytes].copy_(buffer[begin_idx:])
                    if self.remained_not_offloaded is None:
                        raise AssertionError("missing non-offloaded tensor tail during activation reload")
                    linear_data[restored_bytes:].copy_(self.remained_not_offloaded)
                    self.remained_not_offloaded = None
                    restored_tensors[tensor_key] = linear_data
                else:
                    tensor.x = restored_tensors[tensor_key].view(tensor.dtype).view(tensor.shape)
            tensor.activation_group = None
        buffer_base = getattr(buffer, "_base", None)
        if not ping_pong_onload or buffer_base is None or buffer_base.ref_cnt == 0:
            _release_persistent_gpu_buffer(buffer_base)
        self.onload_buffer = None
        self.onload_stream = None
        del self.tensors
        del self.map
        debug_log("onload_epilogue_end", key=self.key)

    def ensure_onloaded(self):
        if self._onloaded:
            return
        args = self.onload_prologue(
            overlap_d2h_h2d=True,
            ping_pong_onload=True,
            onload_buffer_key=f"onload_layer_{id(self)}",
        )
        if args[0] is None:
            return
        for group_id in range(self.offload_group_num):
            self.onload_issue(group_id)
        self.onload_epilogue(*args)
        self._onloaded = True


groups = {}
offload_tensors = None


@dataclass
class _LayerOffloadSession:
    key: object
    group_num: int
    remaining_bytes: int


_layer_offload_session = None


def _tensor_can_offload(tensor_wrap, op_name, require_patch_flag=True):
    if require_patch_flag and not _patch_te_enabled():
        return False
    if op_name not in _offload_modules():
        return False
    x = tensor_wrap.x
    if isinstance(x, torch.nn.Parameter):
        return False
    if x.device.type != "cuda":
        return False
    allow_non_contiguous = bool(
        _get_arg("offload_non_contiguous", _get_arg("bgl2_offload_non_contiguous", False))
    )
    if not allow_non_contiguous:
        if not x.is_contiguous() or x._base is not None or x.storage_offset() != 0:
            return False
    if x.numel() * x.element_size() < _offload_min_bytes():
        return False
    return not (x.dim() == 4 and x.shape[1] == 1 and x.shape[2] == 1)


def pack_hook(x, op_name=None):
    global offload_tensors
    if x is None:
        return None

    force_fp8_ctx = bool(_get_arg("force_fp8_ctx", False))
    fp8_ctx_modules = _get_arg("fp8_ctx_modules", None)
    enable_fp8_ctx = (
        force_fp8_ctx
        and isinstance(x, torch.Tensor)
        and x.dtype == torch.bfloat16
        and fp8_ctx_modules is not None
        and op_name in fp8_ctx_modules
    )

    if fp8_ctx_modules is not None and op_name not in fp8_ctx_modules:
        warnings.warn(f"skip fp8 ctx for {op_name=}")

    if enable_fp8_ctx:
        if has_deep_gemm:
            x_shape = x.shape
            x, scales = fp8_quant_per_tile(x.view(-1, x.shape[-1]))
            tensor_wrap = TensorWrap(x.view(x_shape), scales)
        else:
            if not hasattr(torch, "float8_e4m3fn"):
                raise RuntimeError("force_fp8_ctx requires torch.float8_e4m3fn or deep_gemm")
            tensor_wrap = TensorWrap(x.to(torch.float8_e4m3fn))
    else:
        tensor_wrap = TensorWrap(x)

    if offload_tensors is not None and _tensor_can_offload(tensor_wrap, op_name, require_patch_flag=True):
        offload_tensors.append(tensor_wrap)
    return TensorPack(tensor_wrap, op_name=op_name)


def unpack_hook(tensor_pack):
    if tensor_pack is None:
        return None

    x = tensor_pack.get()
    release_after_unpack = tensor_pack.tensor_wrap.release_after_unpack
    try:
        if (
            bool(_get_arg("force_fp8_ctx", False))
            and hasattr(torch, "float8_e4m3fn")
            and x.dtype == torch.float8_e4m3fn
        ):
            if has_deep_gemm:
                scales = tensor_pack.get_scales()
                x_shape = x.shape
                x = fp8_dequant_per_tile(x.view(-1, x.shape[-1]), scales)
                x = x.view(x_shape)
            else:
                x = x.to(torch.bfloat16)
        return x
    finally:
        if release_after_unpack:
            tensor_pack.release()


def pack_hook_offload(x, op_name=None):
    if x is None:
        return None

    tensor_wrap = TensorWrap(x)
    if offload_tensors is not None and _tensor_can_offload(tensor_wrap, op_name, require_patch_flag=False):
        offload_tensors.append(tensor_wrap)
    return TensorPack(tensor_wrap)


def unpack_hook_offload(tensor_pack):
    if tensor_pack is None:
        return None
    x = tensor_pack.get()
    if tensor_pack.tensor_wrap.release_after_unpack:
        tensor_pack.release()
    return x


@contextlib.contextmanager
def record(key, group_num=1):
    global offload_tensors
    global _layer_offload_session
    if not _patch_te_enabled():
        yield
        return

    offload_ratio = _activation_offload_ratio()
    offload_threshold = _activation_offload_threshold()

    previous = offload_tensors
    previous_layer_session = _layer_offload_session
    if offload_ratio == 0.0 and offload_threshold is None:
        offload_tensors = None
        try:
            yield
        finally:
            groups[key] = ActivationGroup([], key, group_num)
            offload_tensors = previous
    else:
        offload_tensors = []
        if _layer_offload_enabled():
            _layer_offload_session = _LayerOffloadSession(
                key=key,
                group_num=group_num,
                remaining_bytes=_configured_offload_limit_bytes(),
            )
        try:
            yield
        finally:
            offload_limit_bytes = None
            if _layer_offload_session is not previous_layer_session:
                offload_limit_bytes = _layer_offload_session.remaining_bytes
            groups[key] = ActivationGroup(
                offload_tensors,
                key,
                group_num,
                offload_limit_bytes=offload_limit_bytes,
            )
            offload_tensors = previous
            _layer_offload_session = previous_layer_session


@contextlib.contextmanager
def mlp_activation_offload_scope(layer_number):
    global offload_tensors

    session = _layer_offload_session
    if session is None or offload_tensors is None or session.remaining_bytes <= 0:
        yield
        return

    previous = offload_tensors
    offload_tensors = []
    completed = False
    try:
        yield
        completed = True
    finally:
        tensors = offload_tensors
        offload_tensors = previous

    if not completed:
        return

    key = (session.key, "mlp", layer_number)
    group = ActivationGroup(
        tensors,
        key,
        session.group_num,
        offload_limit_bytes=session.remaining_bytes,
        release_source_storage=False,
    )
    offloaded_bytes = group.offload_now()
    session.remaining_bytes = max(0, session.remaining_bytes - offloaded_bytes)


@contextlib.contextmanager
def offload_async(key):
    group = groups[key]
    args = group.offload_prologue(use_bucket=False)
    yield
    group.offload_epilogue(*args)


warn_offload_once = False


class OffloadAsync:
    def __init__(self, key, group_num=1):
        self.disable_offload = not _patch_te_enabled() or key not in groups
        if self.disable_offload:
            debug_log("offload_ctx_disabled", key=key)
            return
        self.issued_group = 0
        self.key = key
        self.group_num = max(1, group_num)
        self.group = groups[self.key]

    def __enter__(self):
        if self.disable_offload:
            return self
        debug_log("offload_ctx_enter_begin", key=self.key, group_num=self.group_num)
        self.group.offload_prologue(use_bucket=False)
        debug_log("offload_ctx_enter_end", key=self.key, group_num=self.group_num)
        return self

    def issue(self, group_id):
        if self.disable_offload:
            return
        assert group_id < self.group_num
        debug_log("offload_ctx_issue_until", key=self.key, group_id=group_id)
        while self.issued_group <= group_id:
            self.group.offload_issue(self.issued_group)
            self.issued_group += 1

    def __exit__(self, *exc_info):
        if self.disable_offload:
            return False
        debug_log("offload_ctx_exit_begin", key=self.key, issued_group=self.issued_group)
        while self.issued_group < self.group_num:
            self.group.offload_issue(self.issued_group)
            self.issued_group += 1
            global warn_offload_once
            if not warn_offload_once:
                warn_offload_once = True
                print("Warning: not all offload groups are issued. Issued at exit.")
        self.group.offload_epilogue()
        debug_log("offload_ctx_exit_end", key=self.key)
        return False


@contextlib.contextmanager
def onload_async(key):
    group = groups[key]
    args = group.onload_prologue(overlap_d2h_h2d=True, ping_pong_onload=True)
    yield
    group.onload_epilogue(*args)


warn_onload_once = False


class OnloadAsync:
    def __init__(self, key, group_num=1):
        self.disable_offload = not _patch_te_enabled() or key not in groups
        if self.disable_offload:
            debug_log("onload_ctx_disabled", key=key)
            return
        self.issued_group = 0
        self.group_num = max(1, group_num)
        self.key = key
        self.group = groups[key]

    def __enter__(self):
        if self.disable_offload:
            return self
        debug_log("onload_ctx_enter_begin", key=self.key, group_num=self.group_num)
        self.args = self.group.onload_prologue(overlap_d2h_h2d=True, ping_pong_onload=True)
        debug_log("onload_ctx_enter_end", key=self.key, group_num=self.group_num)
        return self

    def issue(self, group_id):
        if self.disable_offload:
            return
        assert group_id < self.group_num
        debug_log("onload_ctx_issue_until", key=self.key, group_id=group_id)
        while self.issued_group <= group_id:
            self.group.onload_issue(self.issued_group)
            self.issued_group += 1

    def __exit__(self, *exc_info):
        if self.disable_offload:
            return False
        debug_log("onload_ctx_exit_begin", key=self.key, issued_group=self.issued_group)
        while self.issued_group < self.group_num:
            self.group.onload_issue(self.issued_group)
            self.issued_group += 1
            global warn_onload_once
            if not warn_onload_once:
                warn_onload_once = True
                print("Warning: not all onload groups are issued. Issued at exit.")
        self.group.onload_epilogue(*self.args)
        groups.pop(self.key, None)
        self.args = None
        self.group = None
        debug_log("onload_ctx_exit_end", key=self.key)
        return False


def get_offload_nstages():
    return _activation_offload_stages()


offload_ctx = None
reload_ctx = None


def issue_loads(stage):
    global offload_ctx
    global reload_ctx

    if not _patch_te_enabled():
        return
    assignment = _get_arg("activation_offload_stages_assignment", None)
    if assignment is None:
        assignment = _get_arg("bgl2_activation_offload_stages_assignment", None)
    offload_stage = assignment[stage] if assignment is not None else stage
    debug_log("issue_loads", stage=stage, offload_stage=offload_stage)

    if reload_ctx is not None:
        reload_ctx.issue(offload_stage)
    if offload_ctx is not None:
        offload_ctx.issue(offload_stage)


@dataclass
class _ForwardMicrobatchContext:
    key: object
    group_num: int
    record_ctx: object = None

    def __enter__(self):
        set_current_forward_batch_id(self.key)
        self.record_ctx = record(self.key, self.group_num)
        self.record_ctx.__enter__()
        return self

    def __exit__(self, *exc_info):
        try:
            self.record_ctx.__exit__(*exc_info)
            if exc_info[0] is None:
                offload = OffloadAsync(self.key, self.group_num)
                offload.__enter__()
                offload.__exit__(None, None, None)
            else:
                _OFFLOAD_MANAGER.remove_batch(self.key)
        finally:
            clear_current_forward_batch_id()
        return False


@dataclass
class _BackwardMicrobatchContext:
    key: object
    group_num: int

    def __enter__(self):
        set_current_backward_batch_id(self.key)
        self.onload = OnloadAsync(self.key, self.group_num)
        self.onload.__enter__()
        self.onload.__exit__(None, None, None)
        return self

    def __exit__(self, *exc_info):
        try:
            if exc_info[0] is None:
                remove_batch_from_offload_manager(self.key)
        finally:
            clear_current_backward_batch_id()
        return False


@dataclass
class _InterleavedBackwardMicrobatchContext:
    key: object
    group_num: int

    def __enter__(self):
        if self.key in groups:
            if offload_ctx is not None and getattr(offload_ctx, "key", None) == self.key:
                _flush_offload_ctx(None, None, None)
            onload = OnloadAsync(self.key, self.group_num)
            onload.__enter__()
            onload.issue(self.group_num - 1)
            onload.__exit__(None, None, None)
        return self

    def __exit__(self, *exc_info):
        return False


def _flush_reload_ctx(*exc_info):
    global reload_ctx
    if reload_ctx is None:
        return
    debug_log("flush_reload_begin", key=getattr(reload_ctx, "key", None))
    reload_ctx.__exit__(*exc_info)
    debug_log("flush_reload_end", key=getattr(reload_ctx, "key", None))
    reload_ctx = None


def _flush_offload_ctx(*exc_info):
    global offload_ctx
    if offload_ctx is None:
        return
    debug_log("flush_offload_begin", key=getattr(offload_ctx, "key", None))
    offload_ctx.__exit__(*exc_info)
    debug_log("flush_offload_end", key=getattr(offload_ctx, "key", None))
    offload_ctx = None


def _start_reload_ctx(key, group_num):
    global reload_ctx
    debug_log("start_reload_begin", key=key, group_num=group_num)
    reload_ctx = OnloadAsync(key, group_num)
    reload_ctx.__enter__()
    reload_ctx.issue(group_num - 1)
    debug_log("start_reload_end", key=key, group_num=group_num)


def _start_offload_ctx(key, group_num):
    global offload_ctx
    debug_log("start_offload_begin", key=key, group_num=group_num)
    offload_ctx = OffloadAsync(key, group_num)
    offload_ctx.__enter__()
    offload_ctx.issue(group_num - 1)
    debug_log("start_offload_end", key=key, group_num=group_num)


@dataclass
class _InterleavedMicrobatchStepContext:
    forward_key: object = None
    backward_key: object = None
    reload_key: object = None
    group_num: int = 1
    forward_enabled: bool = True
    reload_enabled: bool = True
    record_ctx: object = None

    def __enter__(self):
        debug_log(
            "interleaved_step_enter_begin",
            forward_key=self.forward_key,
            backward_key=self.backward_key,
            reload_key=self.reload_key,
            forward_enabled=self.forward_enabled,
            reload_enabled=self.reload_enabled,
            group_num=self.group_num,
        )
        _flush_reload_ctx(None, None, None)

        if (
            self.reload_key is not None
            and self.reload_enabled
            and _prefetch_activation_reload()
        ):
            _start_reload_ctx(self.reload_key, self.group_num)

        if self.forward_key is not None:
            set_current_forward_batch_id(self.forward_key)
            if self.forward_enabled:
                self.record_ctx = record(self.forward_key, self.group_num)
                self.record_ctx.__enter__()

        if self.backward_key is not None:
            set_current_backward_batch_id(self.backward_key)

        debug_log(
            "interleaved_step_enter_end",
            forward_key=self.forward_key,
            backward_key=self.backward_key,
            reload_key=self.reload_key,
        )
        return self

    def __exit__(self, *exc_info):
        debug_log(
            "interleaved_step_exit_begin",
            forward_key=self.forward_key,
            backward_key=self.backward_key,
            reload_key=self.reload_key,
            exc=exc_info[0],
        )
        if self.backward_key is not None:
            remove_batch_from_offload_manager()

        _flush_offload_ctx(*exc_info)

        if self.record_ctx is not None:
            self.record_ctx.__exit__(*exc_info)

        if exc_info[0] is None and self.forward_key is not None and self.forward_enabled:
            _start_offload_ctx(self.forward_key, self.group_num)
        if self.forward_key is not None:
            clear_current_forward_batch_id()

        debug_log(
            "interleaved_step_exit_end",
            forward_key=self.forward_key,
            backward_key=self.backward_key,
            reload_key=self.reload_key,
        )
        return False


class PipelineActivationOffloadRuntime:
    def enabled(self):
        return _patch_te_enabled()

    def forward_microbatch(
        self,
        *,
        phase,
        virtual_microbatch_id,
        model_chunk_id,
        enabled=True,
        offload_key=None,
    ):
        if not self.enabled() or not enabled:
            return contextlib.nullcontext()
        if phase == "interleaved":
            return contextlib.nullcontext()
        return _ForwardMicrobatchContext(
            offload_key
            if offload_key is not None
            else _make_key(virtual_microbatch_id, model_chunk_id),
            _activation_offload_stages(),
        )

    def backward_microbatch(
        self,
        *,
        phase,
        virtual_microbatch_id,
        model_chunk_id,
        enabled=True,
        offload_key=None,
    ):
        if not self.enabled() or not enabled:
            return contextlib.nullcontext()
        if phase == "interleaved":
            return _InterleavedBackwardMicrobatchContext(
                offload_key
                if offload_key is not None
                else _make_key(virtual_microbatch_id, model_chunk_id),
                _activation_offload_stages(),
            )
        return _BackwardMicrobatchContext(
            offload_key
            if offload_key is not None
            else _make_key(virtual_microbatch_id, model_chunk_id),
            _activation_offload_stages(),
        )

    def interleaved_step(
        self,
        *,
        forward_key=None,
        backward_key=None,
        reload_key=None,
        forward_enabled=True,
        reload_enabled=True,
    ):
        if not self.enabled():
            return contextlib.nullcontext()
        return _InterleavedMicrobatchStepContext(
            forward_key=forward_key,
            backward_key=backward_key,
            reload_key=reload_key,
            group_num=_activation_offload_stages(),
            forward_enabled=forward_enabled,
            reload_enabled=reload_enabled,
        )

    def flush_interleaved(self):
        _flush_reload_ctx(None, None, None)
        _flush_offload_ctx(None, None, None)


_RUNTIME = PipelineActivationOffloadRuntime()


def get_pipeline_offload_runtime():
    return _RUNTIME
