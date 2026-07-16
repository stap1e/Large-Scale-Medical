"""Loader entry points with lazy imports.

Historically this package imported every dataset module here.  Importing any
``utils.*`` module consequently parsed all abdomen, head/neck, and chest JSON
files.  Keep the public loader names, but resolve them only when requested so a
CT-RATE-only run never initializes unrelated datasets.
"""

from importlib import import_module


_LOADERS = {
    "get_loader_abdomen": ("utils.data_utils_abdomen", "get_loader_abdomen"),
    "get_loader_chest": ("utils.data_utils_chest", "get_loader_chest"),
    "get_loader_headneck": ("utils.data_utils_headneck", "get_loader_headneck"),
    "get_loader": ("utils.data_utils", "get_loader"),
}

__all__ = list(_LOADERS)


def __getattr__(name):
    if name not in _LOADERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LOADERS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
