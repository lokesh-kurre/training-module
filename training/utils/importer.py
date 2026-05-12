"""Dynamic module and object loading utilities inspired by StyleGAN3's dnnlib.util."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any, Tuple


def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """Search for the module containing the python object with the given name.

    Args:
        obj_name: Fully-qualified object name (e.g., "module.submodule.ClassName")

    Returns:
        Tuple of (module, local_obj_name) where local_obj_name is the part after
        the module path (e.g., "training.data.dataset.build_synthetic_dataset" ->
        ("training.data.dataset", "build_synthetic_dataset"))

    Raises:
        ImportError: If the module cannot be found or the object doesn't exist.
    """
    # Split into possible (module_name, local_obj_name) pairs
    parts = obj_name.split(".")
    name_pairs = [((".".join(parts[:i]), ".".join(parts[i:]))) for i in range(len(parts), 0, -1)]

    # Try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)
            get_obj_from_module(module, local_obj_name)  # Verify object exists
            return module, local_obj_name
        except (ImportError, AttributeError):
            pass

    # No luck; raise ImportError with the original name
    raise ImportError(f"Cannot find module for '{obj_name}'")


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """Traverse a nested object path within a module and return the final object.

    Args:
        module: The Python module to search
        obj_name: Dot-separated object path (e.g., "submodule.ClassName.method")

    Returns:
        The requested Python object (function, class, etc.)

    Raises:
        AttributeError: If any part of the path doesn't exist.
    """
    if obj_name == "":
        return module

    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """Find and return a Python object by its fully-qualified name.

    Supports any Python object accessible via module imports:
    - Functions: "module.function_name"
    - Classes: "module.ClassName"
    - Nested objects: "module.Class.nested_attr"

    Args:
        name: Fully-qualified object name

    Returns:
        The requested Python object

    Raises:
        ImportError: If the module or object cannot be found
    """
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """Find a function by name and call it with the given arguments.

    Args:
        func_name: Fully-qualified function name
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Return value of the function call

    Raises:
        ImportError: If the function cannot be found
        TypeError: If the object is not callable
    """
    assert func_name is not None, "func_name must be provided"
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj), f"Object '{func_name}' is not callable"
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """Find a class by name and construct it with the given arguments.

    Args:
        class_name: Fully-qualified class name
        *args: Positional arguments to pass to __init__
        **kwargs: Keyword arguments to pass to __init__

    Returns:
        Instance of the requested class

    Raises:
        ImportError: If the class cannot be found
        TypeError: If the object is not a class
    """
    return call_func_by_name(*args, func_name=class_name, **kwargs)
