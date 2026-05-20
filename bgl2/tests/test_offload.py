import importlib
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock


def _install_fake_megatron_args(state):
    for name in list(sys.modules):
        if name == "megatron" or name.startswith("megatron."):
            del sys.modules[name]

    megatron = types.ModuleType("megatron")
    training = types.ModuleType("megatron.training")

    def get_args():
        return state

    training.get_args = get_args
    megatron.training = training
    sys.modules["megatron"] = megatron
    sys.modules["megatron.training"] = training


def _import_offload_with_fake_torch(state):
    _install_fake_megatron_args(state)
    original_torch = sys.modules.get("torch")
    fake_torch = types.ModuleType("torch")
    fake_torch.uint8 = object()
    sys.modules["torch"] = fake_torch
    sys.modules.pop("bgl2.core.pipeline_parallel.offload", None)
    try:
        return importlib.import_module("bgl2.core.pipeline_parallel.offload"), original_torch
    except Exception:
        if original_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = original_torch
        raise


def _restore_fake_torch(original_torch):
    sys.modules.pop("bgl2.core.pipeline_parallel.offload", None)
    if original_torch is None:
        sys.modules.pop("torch", None)
    else:
        sys.modules["torch"] = original_torch


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class Bgl2OffloadRuntimeTest(unittest.TestCase):
    def test_dcu_offload_manager_tracks_resident_batch_memory(self):
        state = types.SimpleNamespace()
        offload, original_torch = _import_offload_with_fake_torch(state)

        try:
            manager = offload.OffloadManager()
            manager.reset(num_warmup_batches=1, memory_limit_MB=600 / 2**20)

            manager.add_batch(700, "batch")

            self.assertEqual(manager.memory_consumption, 600)
            self.assertAlmostEqual(manager.get_offload_ratio(), 100 / 700)
            manager.remove_batch("batch")
            self.assertEqual(manager.memory_consumption, 0)
        finally:
            _restore_fake_torch(original_torch)

    def test_dynamic_offload_observes_each_forward_batch_once(self):
        state = types.SimpleNamespace()
        offload, original_torch = _import_offload_with_fake_torch(state)

        try:
            offload.reset_offload_manager(
                num_warmup_batches=1,
                memory_limit_MB=600 / 2**20,
            )
            offload.set_current_forward_batch_id("batch")

            offload.add_batch_to_offload_manager(700)
            offload.add_batch_to_offload_manager(700)

            self.assertEqual(offload._OFFLOAD_MANAGER.num_active_batch, 1)
            self.assertEqual(offload._OFFLOAD_MANAGER.memory_consumption, 600)
            self.assertNotIn(None, offload._OFFLOAD_MANAGER.offload_sizes)
        finally:
            _restore_fake_torch(original_torch)

    def test_dynamic_offload_ignores_unobserved_batch_removal(self):
        state = types.SimpleNamespace()
        offload, original_torch = _import_offload_with_fake_torch(state)

        try:
            offload.reset_offload_manager(
                num_warmup_batches=1,
                memory_limit_MB=600 / 2**20,
            )

            offload.remove_batch_from_offload_manager("non-moe-batch")

            self.assertEqual(offload._OFFLOAD_MANAGER.num_active_batch, 0)
            self.assertEqual(offload._OFFLOAD_MANAGER.memory_consumption, 0)
        finally:
            _restore_fake_torch(original_torch)

    def test_non_interleaved_context_tracks_batch_lifetime(self):
        state = types.SimpleNamespace()
        offload, original_torch = _import_offload_with_fake_torch(state)
        events = []

        class FakeContext:
            def __enter__(self):
                events.append(("record-enter",))

            def __exit__(self, *exc_info):
                events.append(("record-exit",))

        class FakeCopy(FakeContext):
            pass

        try:
            offload.record = lambda *args: FakeContext()
            offload.OffloadAsync = lambda *args: FakeCopy()
            offload.OnloadAsync = lambda *args: FakeCopy()
            offload.set_current_forward_batch_id = lambda key: events.append(("set-forward", key))
            offload.set_current_backward_batch_id = lambda key: events.append(("set-backward", key))
            offload.remove_batch_from_offload_manager = lambda key=None: events.append(("remove", key))

            with offload._ForwardMicrobatchContext("batch", 1):
                events.append(("forward",))
            with offload._BackwardMicrobatchContext("batch", 1):
                events.append(("backward",))

            self.assertIn(("set-forward", "batch"), events)
            self.assertIn(("set-backward", "batch"), events)
            self.assertIn(("remove", "batch"), events)
        finally:
            _restore_fake_torch(original_torch)

    def test_cpu_buffer_pool_reuses_smallest_sufficient_capacity(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_offload_pin_memory=True,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)

        class FakeBuffer:
            def __init__(self, size, dtype, base=None):
                self.size = size
                self.dtype = dtype
                self._base = base

            def numel(self):
                return self.size

            def __getitem__(self, item):
                if not isinstance(item, slice) or item.start not in (None, 0):
                    raise TypeError("fake buffer only supports slices from zero")
                stop = self.size if item.stop is None else item.stop
                root = self if self._base is None else self._base
                return FakeBuffer(stop, self.dtype, base=root)

        allocations = []

        def fake_empty(size, *, dtype, pin_memory):
            self.assertTrue(pin_memory)
            buffer = FakeBuffer(size, dtype)
            allocations.append(buffer)
            return buffer

        try:
            offload.torch.empty = fake_empty
            dtype = offload.torch.uint8

            first = offload.get_cpu_buffer(128, dtype=dtype)
            offload.recycle_cpu_buffer(first)
            second = offload.get_cpu_buffer(96, dtype=dtype)

            self.assertEqual(len(allocations), 1)
            self.assertEqual(second.numel(), 96)
            self.assertIs(second._base, allocations[0])

            offload.recycle_cpu_buffer(second)
            third = offload.get_cpu_buffer(128, dtype=dtype)
            self.assertEqual(len(allocations), 1)
            self.assertEqual(third.numel(), 128)
            self.assertIs(third._base, allocations[0])
        finally:
            _restore_fake_torch(original_torch)

    def test_cpu_buffer_pool_uses_best_fit_buffer(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_offload_pin_memory=True,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)

        class FakeBuffer:
            def __init__(self, size, dtype, base=None):
                self.size = size
                self.dtype = dtype
                self._base = base

            def numel(self):
                return self.size

            def __getitem__(self, item):
                stop = self.size if item.stop is None else item.stop
                root = self if self._base is None else self._base
                return FakeBuffer(stop, self.dtype, base=root)

        try:
            dtype = offload.torch.uint8
            smaller = FakeBuffer(112, dtype)
            larger = FakeBuffer(128, dtype)
            offload._PINNED_BUFFER_POOL[112, dtype].append(smaller)
            offload._PINNED_BUFFER_POOL[128, dtype].append(larger)
            offload.torch.empty = lambda *args, **kwargs: self.fail(
                "a sufficient pooled buffer should prevent allocation"
            )

            result = offload.get_cpu_buffer(96, dtype=dtype)

            self.assertIs(result._base, smaller)
            self.assertEqual(len(offload._PINNED_BUFFER_POOL[128, dtype]), 1)
        finally:
            _restore_fake_torch(original_torch)

    def test_cpu_buffer_pool_replaces_smallest_insufficient_buffer(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_offload_pin_memory=True,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)

        class FakeBuffer:
            def __init__(self, size, dtype, base=None):
                self.size = size
                self.dtype = dtype
                self._base = base

            def numel(self):
                return self.size

            def __getitem__(self, item):
                stop = self.size if item.stop is None else item.stop
                root = self if self._base is None else self._base
                return FakeBuffer(stop, self.dtype, base=root)

        allocations = []

        def fake_empty(size, *, dtype, pin_memory):
            buffer = FakeBuffer(size, dtype)
            allocations.append(buffer)
            return buffer

        try:
            dtype = offload.torch.uint8
            granularity = offload._PINNED_BUFFER_GRANULARITY_BYTES
            smaller = FakeBuffer(granularity - 1, dtype)
            larger = FakeBuffer(granularity, dtype)
            offload._PINNED_BUFFER_POOL[smaller.numel(), dtype].append(smaller)
            offload._PINNED_BUFFER_POOL[larger.numel(), dtype].append(larger)
            offload.torch.empty = fake_empty

            requested = 2 * granularity
            result = offload.get_cpu_buffer(requested, dtype=dtype)

            self.assertEqual(result.numel(), requested)
            self.assertEqual(allocations[0].numel(), 3 * granularity)
            self.assertNotIn((smaller.numel(), dtype), offload._PINNED_BUFFER_POOL)
            self.assertEqual(len(offload._PINNED_BUFFER_POOL[larger.numel(), dtype]), 1)
            self.assertEqual(offload._PINNED_BUFFER_POOL_STATS["replacements"], 1)

            offload.recycle_cpu_buffer(result)
            pooled_buffers = sum(
                len(buffers) for buffers in offload._PINNED_BUFFER_POOL.values()
            )
            self.assertEqual(pooled_buffers, 2)

            first_active = offload.get_cpu_buffer(granularity // 2, dtype=dtype)
            second_requested = 3 * granularity + 1
            second_active = offload.get_cpu_buffer(second_requested, dtype=dtype)
            self.assertEqual(second_active.numel(), second_requested)
            self.assertEqual(second_active._base.numel(), 4 * granularity)

            offload.recycle_cpu_buffer(first_active)
            offload.recycle_cpu_buffer(second_active)
            pooled_buffers = sum(
                len(buffers) for buffers in offload._PINNED_BUFFER_POOL.values()
            )
            self.assertEqual(pooled_buffers, 2)
            self.assertEqual(offload._PINNED_BUFFER_POOL_STATS["replacements"], 2)
        finally:
            _restore_fake_torch(original_torch)

    def test_disabled_memory_reporter_is_a_noop(self):
        state = types.SimpleNamespace()
        offload, original_torch = _import_offload_with_fake_torch(state)

        try:
            reporter = offload.MemoryReporter([0], enabled=False)
            reporter.reset()
            reporter.record()
            reporter.report()

            self.assertFalse(reporter.enabled)
            self.assertFalse(reporter.print)
            self.assertEqual(reporter.base_mem, 0)
            self.assertEqual(reporter.max_mem, 0)
        finally:
            _restore_fake_torch(original_torch)

    def test_activation_group_ignores_packs_released_during_no_grad_forward(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)
        released_wrap = types.SimpleNamespace(x=None)

        try:
            group = offload.ActivationGroup([released_wrap], key="eval-layer", offload_groups=1)

            self.assertEqual(group.tensors, [])
        finally:
            _restore_fake_torch(original_torch)

    def test_layer_group_releases_references_without_resizing_shared_storage(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)

        class FakeStream:
            def wait_stream(self, _stream):
                pass

        class FakeBuffer:
            def numel(self):
                return 4

        class FakeStorage:
            def resize_(self, _size):
                raise AssertionError(
                    "layer-scoped offload must not resize aliased source storage"
                )

        class FakeTensor:
            def untyped_storage(self):
                return FakeStorage()

        source = FakeTensor()
        tensor_wrap = types.SimpleNamespace(x=None, outer=source)

        try:
            offload.torch.cuda = types.SimpleNamespace(current_stream=lambda: FakeStream())
            offload.get_memcpy_stream = lambda _key: FakeStream()
            group = offload.ActivationGroup([], key="layer", offload_groups=1)
            group.tensors = [tensor_wrap]
            group.offload_groups = [[(0, 4, source)]]
            group.buffer_cpu = FakeBuffer()
            group.release_source_storage = False

            group.offload_epilogue()

            self.assertIsNone(tensor_wrap.outer)
            self.assertEqual(group.offload_groups, [[]])
        finally:
            _restore_fake_torch(original_torch)

    def test_layer_scope_offloads_before_microbatch_exit_and_reloads_lazily(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_offload_granularity="layer",
            bgl2_activation_offload_stages=2,
            bgl2_per_batch_offload_size=1,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)
        events = []

        class FakeGroup:
            def __init__(
                self,
                tensors,
                key,
                offload_groups,
                *,
                offload_limit_bytes=None,
                release_source_storage=True,
            ):
                self.tensors = tensors
                self.key = key
                self.offload_groups = offload_groups
                self.offload_limit_bytes = offload_limit_bytes
                self.release_source_storage = release_source_storage
                self.reloads = 0

            def offload_now(self):
                events.append(("offload", self.key, len(self.tensors)))
                for tensor in self.tensors:
                    tensor.activation_group = self
                    tensor.x = None
                return 64

            def ensure_onloaded(self):
                self.reloads += 1
                events.append(("reload", self.key))
                for tensor in self.tensors:
                    tensor.x = "restored"
                    tensor.activation_group = None

        tensor_wrap = types.SimpleNamespace(
            x="cuda",
            outer="cuda",
            scales=None,
            base=None,
            activation_group=None,
        )

        try:
            offload.ActivationGroup = FakeGroup
            offload.groups.clear()

            with offload.record("microbatch-0", group_num=2):
                with offload.mlp_activation_offload_scope(layer_number=7):
                    offload.offload_tensors.append(tensor_wrap)
                events.append(("next-layer",))

            self.assertEqual(
                events[:2],
                [
                    ("offload", ("microbatch-0", "mlp", 7), 1),
                    ("next-layer",),
                ],
            )
            self.assertEqual(offload.groups["microbatch-0"].tensors, [])
            self.assertEqual(
                offload.groups["microbatch-0"].offload_limit_bytes,
                1024 * 1024 - 64,
            )

            tensor_pack = offload.TensorPack(tensor_wrap)
            self.assertEqual(tensor_pack.get(), "restored")
            self.assertEqual(tensor_pack.get(), "restored")
            self.assertEqual(events.count(("reload", ("microbatch-0", "mlp", 7))), 1)
        finally:
            offload.groups.clear()
            _restore_fake_torch(original_torch)

    def test_debug_memory_log_includes_cuda_memory_fields(self):
        state = types.SimpleNamespace()
        offload, original_torch = _import_offload_with_fake_torch(state)

        try:
            offload.torch.cuda = types.SimpleNamespace(
                is_available=lambda: True,
                memory_allocated=lambda: 1024,
                memory_reserved=lambda: 2048,
                max_memory_allocated=lambda: 4096,
            )
            with mock.patch.dict(
                os.environ,
                {"BGL2_OFFLOAD_DEBUG": "1", "RANK": "0", "LOCAL_RANK": "0"},
            ):
                with mock.patch("builtins.print") as print_mock:
                    offload.debug_memory_log("offload", key="microbatch", bytes=512)

            output = print_mock.call_args.args[0]
            self.assertIn("event=offload", output)
            self.assertIn("bytes=512", output)
            self.assertIn("cuda_allocated_bytes=1024", output)
            self.assertIn("cuda_reserved_bytes=2048", output)
            self.assertIn("cuda_max_allocated_bytes=4096", output)
            self.assertIn("timestamp_ns=", output)
        finally:
            _restore_fake_torch(original_torch)

    def test_interleaved_step_keeps_offload_and_reload_pending_across_compute(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_activation_offload_stages=2,
            bgl2_per_batch_offload_size=0,
            bgl2_prefetch_activation_reload=True,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)
        events = []

        class FakeRecord:
            def __init__(self, key, group_num):
                self.key = key
                self.group_num = group_num

            def __enter__(self):
                events.append(("record_enter", self.key, self.group_num))

            def __exit__(self, *exc_info):
                events.append(("record_exit", self.key))

        class FakeCopy:
            kind = None

            def __init__(self, key, group_num):
                self.key = key
                self.group_num = group_num

            def __enter__(self):
                events.append((f"{self.kind}_enter", self.key, self.group_num))
                return self

            def issue(self, group_id):
                events.append((f"{self.kind}_issue", self.key, group_id))

            def __exit__(self, *exc_info):
                events.append((f"{self.kind}_exit", self.key))

        class FakeOffload(FakeCopy):
            kind = "offload"

        class FakeOnload(FakeCopy):
            kind = "onload"

        try:
            offload.record = lambda key, group_num=1: FakeRecord(key, group_num)
            offload.OffloadAsync = FakeOffload
            offload.OnloadAsync = FakeOnload
            offload.set_current_forward_batch_id = lambda key: events.append(("set_fwd", key))
            offload.set_current_backward_batch_id = lambda key: events.append(("set_bwd", key))
            offload.remove_batch_from_offload_manager = lambda: events.append(("remove_bwd",))
            offload.offload_ctx = None
            offload.reload_ctx = None

            runtime = offload.get_pipeline_offload_runtime()
            with runtime.interleaved_step(forward_key="f0", forward_enabled=True):
                events.append(("compute", "f0"))

            with runtime.interleaved_step(
                forward_key="f1",
                backward_key="b0",
                reload_key="b1",
                forward_enabled=True,
                reload_enabled=True,
            ):
                events.append(("compute", "f1+b0"))

            runtime.flush_interleaved()

            self.assertEqual(
                events,
                [
                    ("set_fwd", "f0"),
                    ("record_enter", "f0", 2),
                    ("compute", "f0"),
                    ("record_exit", "f0"),
                    ("offload_enter", "f0", 2),
                    ("offload_issue", "f0", 1),
                    ("onload_enter", "b1", 2),
                    ("onload_issue", "b1", 1),
                    ("set_fwd", "f1"),
                    ("record_enter", "f1", 2),
                    ("set_bwd", "b0"),
                    ("compute", "f1+b0"),
                    ("remove_bwd",),
                    ("offload_exit", "f0"),
                    ("record_exit", "f1"),
                    ("offload_enter", "f1", 2),
                    ("offload_issue", "f1", 1),
                    ("onload_exit", "b1"),
                    ("offload_exit", "f1"),
                ],
            )
        finally:
            _restore_fake_torch(original_torch)

    def test_interleaved_step_does_not_prefetch_reload_by_default(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_activation_offload_stages=1,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)
        events = []

        class FakeOnload:
            def __init__(self, key, group_num):
                events.append(("onload_init", key, group_num))

            def __enter__(self):
                events.append(("onload_enter",))
                return self

            def issue(self, group_id):
                events.append(("onload_issue", group_id))

            def __exit__(self, *exc_info):
                events.append(("onload_exit",))

        try:
            offload.OnloadAsync = FakeOnload
            offload.reload_ctx = None

            runtime = offload.get_pipeline_offload_runtime()
            with runtime.interleaved_step(reload_key="next-backward"):
                events.append(("compute",))
            runtime.flush_interleaved()

            self.assertEqual(events, [("compute",)])
        finally:
            _restore_fake_torch(original_torch)

    def test_zero_per_batch_offload_size_disables_fixed_budget_offload(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_activation_offload_ratio=1.0,
            bgl2_per_batch_offload_size=0,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)

        try:
            self.assertEqual(offload._offload_size_bytes(1536 * 1024 * 1024, 2), 0)

            state.bgl2_per_batch_offload_size = 512
            self.assertEqual(
                offload._offload_size_bytes(1536 * 1024 * 1024, 2),
                512 * 1024 * 1024,
            )
            self.assertEqual(
                offload._offload_size_bytes(256 * 1024 * 1024, 2),
                256 * 1024 * 1024,
            )
        finally:
            _restore_fake_torch(original_torch)

    def test_onload_async_uses_ping_pong_storage(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)
        events = []

        class FakeGroup:
            def onload_prologue(self, *, overlap_d2h_h2d, ping_pong_onload):
                events.append(("prologue", overlap_d2h_h2d, ping_pong_onload))
                return None, None, ping_pong_onload

            def onload_issue(self, group_id):
                events.append(("issue", group_id))

            def onload_epilogue(self, *args):
                events.append(("epilogue", *args))

        try:
            offload.groups["microbatch"] = FakeGroup()
            onload = offload.OnloadAsync("microbatch", group_num=1)
            onload.__enter__()
            onload.issue(0)
            onload.__exit__(None, None, None)

            self.assertEqual(events[0], ("prologue", True, True))
        finally:
            offload.groups.clear()
            _restore_fake_torch(original_torch)

    def test_full_reload_reuses_duplicate_tensor(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_activation_offload_ratio=1.0,
            bgl2_per_batch_offload_size=1,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)
        clones = []

        class FakeStream:
            def wait_stream(self, _stream):
                pass

        class FakeBuffer:
            dtype = "uint8"

            def __init__(self, base):
                self._base = base

            def numel(self):
                return 4

            def __getitem__(self, _item):
                return FakeBuffer(self._base)

            def view(self, *_args):
                return self

            def clone(self):
                clone = FakeBuffer(None)
                clones.append(clone)
                return clone

        first = types.SimpleNamespace(x=None, shape=4, dtype="uint8", device="cuda", base=None)
        duplicate = types.SimpleNamespace(x=None, shape=4, dtype="uint8", device="cuda", base=None)

        try:
            offload.torch.cuda = types.SimpleNamespace(current_stream=lambda: FakeStream())
            offload.recycle_cpu_buffer = lambda _buffer: None
            group = offload.ActivationGroup([], key="duplicates", offload_groups=1)
            group.tensors = [first, duplicate]
            group.map = [(0, 4, False), (0, 4, True)]
            group.offload_groups = [[("unused",)]]
            group.buffer_cpu = object()

            group.onload_epilogue(
                FakeStream(),
                FakeBuffer(types.SimpleNamespace(ref_cnt=0)),
                ping_pong_onload=False,
            )

            self.assertEqual(len(clones), 1)
            self.assertIs(first.x, duplicate.x)
        finally:
            _restore_fake_torch(original_torch)

    def test_reloaded_buffer_is_unpooled_after_last_saved_tensor_is_released(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)
        base = types.SimpleNamespace(ref_cnt=1)
        tensor_wrap = types.SimpleNamespace(x=object(), scales=None, base=base)

        try:
            offload._GPU_BUFFER_POOL["onload_ping"] = base
            tensor_pack = offload.TensorPack(tensor_wrap)
            tensor_pack.__del__()

            self.assertNotIn("onload_ping", offload._GPU_BUFFER_POOL)
        finally:
            tensor_wrap.base = None
            offload._GPU_BUFFER_POOL.clear()
            _restore_fake_torch(original_torch)

    def test_layer_unpack_releases_pool_ownership_immediately(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            force_fp8_ctx=False,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)
        restored = object()
        base = types.SimpleNamespace(ref_cnt=1)
        tensor_wrap = types.SimpleNamespace(
            x=restored,
            outer=None,
            scales=None,
            base=base,
            activation_group=None,
            release_after_unpack=True,
        )

        try:
            offload._GPU_BUFFER_POOL["onload_layer_test"] = base
            tensor_pack = offload.TensorPack(tensor_wrap)

            self.assertIs(offload.unpack_hook(tensor_pack), restored)
            self.assertIsNone(tensor_wrap.x)
            self.assertIsNone(tensor_wrap.base)
            self.assertEqual(base.ref_cnt, 0)
            self.assertNotIn("onload_layer_test", offload._GPU_BUFFER_POOL)
        finally:
            tensor_wrap.base = None
            offload._GPU_BUFFER_POOL.clear()
            _restore_fake_torch(original_torch)

    def test_non_contiguous_and_view_offload_requires_opt_in(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_offload_modules=["target_op"],
            bgl2_offload_min_bytes=0,
            bgl2_offload_non_contiguous=False,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)

        class FakeParameter:
            pass

        class FakeTensor:
            def __init__(self, *, contiguous=True, base=None, storage_offset=0):
                self.device = types.SimpleNamespace(type="cuda")
                self._contiguous = contiguous
                self._base = base
                self._storage_offset = storage_offset
                self.shape = (8,)

            def numel(self):
                return 8

            def element_size(self):
                return 2

            def is_contiguous(self):
                return self._contiguous

            def storage_offset(self):
                return self._storage_offset

            def dim(self):
                return 1

        try:
            offload.torch.nn = types.SimpleNamespace(Parameter=FakeParameter)
            contiguous = types.SimpleNamespace(x=FakeTensor())
            non_contiguous = types.SimpleNamespace(x=FakeTensor(contiguous=False))
            view = types.SimpleNamespace(x=FakeTensor(base=object()))
            offset = types.SimpleNamespace(x=FakeTensor(storage_offset=1))

            self.assertTrue(offload._tensor_can_offload(contiguous, "target_op"))
            self.assertFalse(offload._tensor_can_offload(non_contiguous, "target_op"))
            self.assertFalse(offload._tensor_can_offload(view, "target_op"))
            self.assertFalse(offload._tensor_can_offload(offset, "target_op"))

            state.bgl2_offload_non_contiguous = True
            self.assertTrue(offload._tensor_can_offload(non_contiguous, "target_op"))
            self.assertTrue(offload._tensor_can_offload(view, "target_op"))
            self.assertTrue(offload._tensor_can_offload(offset, "target_op"))
        finally:
            _restore_fake_torch(original_torch)

    def test_zero_offload_group_does_not_touch_buffers_or_streams(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_activation_offload_ratio=1.0,
            bgl2_per_batch_offload_size=0,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)

        class FakeShape:
            def numel(self):
                return 1536 * 1024 * 1024

        class FakeDtype:
            itemsize = 1

        class FakeTensor:
            shape = FakeShape()
            dtype = FakeDtype()
            device = types.SimpleNamespace(type="cuda")
            _base = None

            def is_contiguous(self):
                return True

            def data_ptr(self):
                return 1

        try:
            offload.print_once = True
            offload.get_cpu_buffer = lambda *args, **kwargs: self.fail(
                "zero offload should not allocate a CPU buffer"
            )
            offload.get_memcpy_stream = lambda *args, **kwargs: self.fail(
                "zero offload should not create or wait on memcpy streams"
            )

            tensor_wrap = types.SimpleNamespace(
                x=FakeTensor(),
                shape=FakeShape(),
                dtype=FakeDtype(),
                device=types.SimpleNamespace(type="cuda"),
                base=None,
            )
            group = offload.ActivationGroup([tensor_wrap], key="microbatch", offload_groups=2)

            self.assertEqual(group.offload_prologue(use_bucket=False), (None, None, None))
            self.assertEqual(group.offload_groups, [])
            group.offload_issue(0)
            group.offload_epilogue()

            onload_args = group.onload_prologue(
                overlap_d2h_h2d=True,
                ping_pong_onload=False,
            )
            self.assertEqual(onload_args, (None, None, False))
            group.onload_issue(0)
            group.onload_epilogue(*onload_args)
        finally:
            _restore_fake_torch(original_torch)

    def test_partial_offload_reload_restores_non_offloaded_tail_without_torch(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_activation_offload_ratio=1.0,
            bgl2_per_batch_offload_size=0,
        )
        offload, original_torch = _import_offload_with_fake_torch(state)

        class FakeStream:
            def wait_stream(self, _stream):
                pass

        class FakeTensor:
            def __init__(self, data, dtype="uint8", shape=None, device="cuda"):
                self.data = data
                self.dtype = dtype
                self.shape = shape or len(data)
                self.device = device

            def numel(self):
                return len(self.data)

            def view(self, *_args):
                return self

            def copy_(self, other):
                self.data[:] = list(other.data)

            def __getitem__(self, item):
                if not isinstance(item, slice):
                    raise TypeError("fake tensor only supports slicing")
                return FakeTensorView(self, item)

        class FakeTensorView(FakeTensor):
            def __init__(self, parent, item):
                self.parent = parent
                self.item = item
                super().__init__(parent.data[item], dtype=parent.dtype, device=parent.device)

            def copy_(self, other):
                self.parent.data[self.item] = list(other.data)
                self.data = self.parent.data[self.item]

        output = []

        def fake_empty(shape, dtype, device):
            tensor = FakeTensor([None] * shape, dtype=dtype, shape=shape, device=device)
            output.append(tensor)
            return tensor

        try:
            offload.torch.cuda = types.SimpleNamespace(current_stream=lambda: FakeStream())
            offload.torch.empty = fake_empty
            offload.recycle_cpu_buffer = lambda _buffer: None

            group = offload.ActivationGroup([], key="partial", offload_groups=1)
            group.tensors = [
                types.SimpleNamespace(
                    x=None,
                    shape=4,
                    dtype="uint8",
                    device="cuda",
                    base=None,
                )
            ]
            group.map = [(0, 4, False)]
            group.offload_groups = [[("unused",)]]
            group.buffer_cpu = FakeTensor([0, 0], dtype="uint8")
            group.remained_not_offloaded = FakeTensor([2, 3], dtype="uint8")

            group.onload_epilogue(FakeStream(), FakeTensor([0, 1], dtype="uint8"), False)

            self.assertEqual(output[0].data, [0, 1, 2, 3])
        finally:
            _restore_fake_torch(original_torch)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for offload hook tests")
class Bgl2OffloadTest(unittest.TestCase):
    def test_pack_hook_records_only_configured_op_names(self):
        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_offload_modules=["target_op"],
            bgl2_offload_min_bytes=0,
            bgl2_activation_offload_ratio=1.0,
            bgl2_per_batch_offload_size=0,
            bgl2_activation_offload_stages=1,
            force_fp8_ctx=False,
            fp8_ctx_modules=None,
        )
        _install_fake_megatron_args(state)
        sys.modules.pop("bgl2.core.pipeline_parallel.offload", None)
        offload = importlib.import_module("bgl2.core.pipeline_parallel.offload")

        class FakeShape(tuple):
            def numel(self):
                return 4

        class FakeCudaTensor:
            shape = FakeShape((4,))
            dtype = types.SimpleNamespace(itemsize=4)
            device = types.SimpleNamespace(type="cuda")
            _base = None

            def is_contiguous(self):
                return True

            def storage_offset(self):
                return 0

            def numel(self):
                return 4

            def element_size(self):
                return 4

            def dim(self):
                return 1

        packs = []
        with offload.record("microbatch", group_num=1):
            packs.append(offload.pack_hook(FakeCudaTensor(), op_name="other_op"))
            packs.append(offload.pack_hook(FakeCudaTensor(), op_name="target_op"))

        group = offload.groups["microbatch"]

        self.assertEqual(len(group.tensors), 1)
        self.assertEqual(packs[1].op_name, "target_op")

    def test_partial_offload_reload_restores_non_offloaded_tail(self):
        import torch

        state = types.SimpleNamespace(
            bgl2_patch_te=True,
            bgl2_enable_activation_offload=True,
            bgl2_offload_modules=["target_op"],
            bgl2_offload_min_bytes=0,
            bgl2_activation_offload_ratio=1.0,
            bgl2_per_batch_offload_size=0,
            bgl2_activation_offload_stages=1,
            force_fp8_ctx=False,
            fp8_ctx_modules=None,
        )
        _install_fake_megatron_args(state)
        sys.modules.pop("bgl2.core.pipeline_parallel.offload", None)
        offload = importlib.import_module("bgl2.core.pipeline_parallel.offload")

        class FakeStream:
            def wait_stream(self, _stream):
                pass

        expected = torch.arange(4, dtype=torch.float32)
        expected_bytes = expected.view(torch.uint8)
        offloaded_bytes = expected_bytes[:8].clone()
        tail_bytes = expected_bytes[8:].clone()

        tensor_wrap = offload.TensorWrap(torch.empty_like(expected))
        group = offload.ActivationGroup([], key="partial", offload_groups=1)
        group.tensors = [tensor_wrap]
        group.map = [(0, expected_bytes.numel(), False)]
        group.offload_groups = [[(0, offloaded_bytes.numel(), offloaded_bytes)]]
        group.buffer_cpu = torch.empty_like(offloaded_bytes)
        group.remained_not_offloaded = tail_bytes

        with mock.patch.object(offload.torch.cuda, "current_stream", return_value=FakeStream()):
            with mock.patch.object(offload, "recycle_cpu_buffer", lambda _buffer: None):
                group.onload_epilogue(FakeStream(), offloaded_bytes, ping_pong_onload=False)

        torch.testing.assert_close(tensor_wrap.x, expected)


if __name__ == "__main__":
    unittest.main()
