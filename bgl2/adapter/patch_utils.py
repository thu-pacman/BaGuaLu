"""Minimal clean-room patch helper used by bgl2.

The helper only supports the patching mode bgl2 needs for Megatron-LM:
wrapping an existing symbol and updating already-imported aliases that still
point at the original object.
"""

import importlib
import sys
from dataclasses import dataclass, field


def _split_target(target_name):
    owner_name, separator, attr_name = target_name.rpartition(".")
    if not separator:
        raise ValueError("patch target must include a module and attribute")
    return owner_name, attr_name


def _load_owner(owner_name):
    return importlib.import_module(owner_name)


def _resolve_target(target_name):
    parts = target_name.split(".")
    if len(parts) < 2:
        raise ValueError("patch target must include a module and attribute")

    import_error = None
    for owner_part_count in range(len(parts) - 1, 0, -1):
        owner_name = ".".join(parts[:owner_part_count])
        attr_path = parts[owner_part_count:]

        try:
            owner = _load_owner(owner_name)
        except ModuleNotFoundError as exc:
            missing_name = exc.name
            if (
                missing_name is not None
                and missing_name != owner_name
                and not owner_name.startswith(f"{missing_name}.")
            ):
                raise
            import_error = exc
            continue

        for attr_name in attr_path[:-1]:
            owner = getattr(owner, attr_name)
        return owner, attr_path[-1]

    if import_error is not None:
        raise import_error
    raise ValueError("patch target must include a module and attribute")


def _replace_matching_aliases(attr_name, original, patched):
    for module in list(sys.modules.values()):
        if module is None or not hasattr(module, "__dict__"):
            continue
        if attr_name not in module.__dict__:
            continue
        if module.__dict__[attr_name] is original:
            setattr(module, attr_name, patched)


@dataclass
class Patch:
    target_name: str
    replacement: object = None
    apply_wrapper: bool = False
    wrappers: list = field(default_factory=list)
    is_applied: bool = False

    def __post_init__(self):
        initial_replacement = self.replacement
        self.replacement = None
        if initial_replacement is not None:
            self.set_patch_func(initial_replacement, apply_wrapper=self.apply_wrapper)

    def set_patch_func(
        self,
        replacement=None,
        force_patch=False,
        apply_wrapper=False,
        remove_origin_wrappers=False,
    ):
        if remove_origin_wrappers:
            raise NotImplementedError("bgl2 patch helper does not strip existing wrappers")
        if replacement is None:
            raise AssertionError("replacement must be provided")

        if apply_wrapper:
            if not any(wrapper is replacement for wrapper in self.wrappers):
                self.wrappers.append(replacement)
                self.is_applied = False
            return

        if self.replacement is not None and self.replacement is not replacement and not force_patch:
            raise RuntimeError("patch for {} already exists".format(self.target_name))
        self.replacement = replacement
        self.is_applied = False

    def apply_patch(self):
        if self.is_applied:
            return

        owner, attr_name = _resolve_target(self.target_name)
        original = getattr(owner, attr_name)
        patched = self.replacement if self.replacement is not None else original

        for wrapper in self.wrappers:
            patched = wrapper(patched)

        setattr(owner, attr_name, patched)
        _replace_matching_aliases(attr_name, original, patched)
        self.is_applied = True


class MegatronPatchesManager:
    patches_info = {}

    @staticmethod
    def register_patch(
        orig_func_or_cls_name,
        new_func_or_cls=None,
        force_patch=False,
        create_dummy=False,
        apply_wrapper=False,
        remove_origin_wrappers=False,
        **kwargs
    ):
        if create_dummy:
            raise NotImplementedError("bgl2 patch helper requires existing Megatron symbols")
        if "new_func" in kwargs and new_func_or_cls is None:
            new_func_or_cls = kwargs.pop("new_func")
        if kwargs:
            raise TypeError("unexpected keyword argument: {}".format(next(iter(kwargs))))

        patch = MegatronPatchesManager.patches_info.get(orig_func_or_cls_name)
        if patch is None:
            patch = Patch(
                orig_func_or_cls_name,
                new_func_or_cls,
                apply_wrapper=apply_wrapper,
            )
            MegatronPatchesManager.patches_info[orig_func_or_cls_name] = patch
            return

        patch.set_patch_func(
            new_func_or_cls,
            force_patch=force_patch,
            apply_wrapper=apply_wrapper,
            remove_origin_wrappers=remove_origin_wrappers,
        )

    @staticmethod
    def apply_patches():
        for patch in MegatronPatchesManager.patches_info.values():
            patch.apply_patch()

    @staticmethod
    def reset():
        MegatronPatchesManager.patches_info = {}
