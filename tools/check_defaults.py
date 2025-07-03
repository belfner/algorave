from __future__ import annotations

import inspect
import sys

import algorave
from algorave.core.transforms_interface import BasicTransform


def check_apply_methods(cls):
    """Check for issues in 'apply' methods related to default arguments and Optional type annotations."""
    issues = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("apply"):
            signature = inspect.signature(method)
            issues.extend(
                f"Default argument found in {cls.__name__}.{name} for parameter "
                f"{param.name} with default value {param.default}"
                for param in signature.parameters.values()
                if param.default is not inspect.Parameter.empty
            )
    return issues


def is_subclass_of_basic_transform(cls):
    """Check if a given class is a subclass of BasicTransform, excluding BasicTransform itself."""
    return issubclass(cls, BasicTransform) and cls is not BasicTransform


def main():
    issues = []
    # Check all classes in the algorave module
    for _name, cls in inspect.getmembers(algorave, predicate=inspect.isclass):
        if is_subclass_of_basic_transform(cls):
            issues.extend(check_apply_methods(cls))

    if issues:
        print("\n".join(issues))
        sys.exit(1)  # Exit with error status 1 if there are any issues


if __name__ == "__main__":
    main()
