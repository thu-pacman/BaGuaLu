
import sys
import logging
import importlib
import importlib.util
from importlib.metadata import version
from packaging.version import Version as PkgVersion

from megatron.core.utils import get_te_version
from transformer_engine.common import load_framework_extension

def _load_library():
    load_framework_extension("torch")

_load_library()

if get_te_version() == PkgVersion("2.12.0"):
    from .v2_12_0.linear import Linear
    from .v2_12_0.grouped_linear import GroupedLinear
    from .v2_12_0.layernorm_linear import LayerNormLinear
    from .v2_12_0.permutation import moe_permute, moe_unpermute, moe_sort_chunks_by_index, _moe_permute_mask_map, _moe_unpermute_mask_map
else:
    raise NotImplementedError(f"{get_te_version()=} is not compatible with fp8 ctx.")
