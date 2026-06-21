"""Dataset utilities for Global SAR Ship-Sea Classification.

A full PyTorch Dataset implementation will be added later. This placeholder
keeps the repository structure clear for beginner contributors.
"""

CLASSES = ("ship", "sea")


def describe_classes() -> str:
    """Return a short description of the binary classification classes."""
    return "Binary SAR classes: ship and sea"
