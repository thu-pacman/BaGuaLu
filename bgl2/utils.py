"""Small runtime helpers for bgl2 entrypoints."""

import os
import sys


def add_python_path(path):
    """Prepend an explicit filesystem path to sys.path and PYTHONPATH."""
    resolved_path = os.path.abspath(os.fspath(path))

    if resolved_path not in sys.path:
        sys.path.insert(0, resolved_path)

    pythonpath = os.environ.get("PYTHONPATH")
    env_paths = pythonpath.split(os.pathsep) if pythonpath else []
    env_paths = [env_path for env_path in env_paths if env_path != resolved_path]
    os.environ["PYTHONPATH"] = os.pathsep.join([resolved_path, *env_paths])

    return resolved_path
