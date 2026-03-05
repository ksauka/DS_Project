"""
Pytest configuration: stub heavyweight ML packages that are not installed
in the test environment so pure-logic unit tests can run without a GPU stack.
"""

import sys
import types
from unittest.mock import MagicMock


def _make_torch_stub():
    """Build a minimal torch stub that satisfies scipy's compatibility checks."""
    torch = types.ModuleType("torch")

    class _Tensor:
        pass

    torch.Tensor = _Tensor
    torch.device = MagicMock
    torch.cuda = MagicMock()
    torch.cuda.is_available = lambda: False
    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Module = MagicMock
    return torch


# Register stubs before any project imports happen
_stubs = {
    "torch": _make_torch_stub(),
    "torch.nn": None,           # placeholder — filled after torch is set
    "sentence_transformers": None,
    "transformers": None,
}

if "torch" not in sys.modules:
    sys.modules["torch"] = _stubs["torch"]
    sys.modules["torch.nn"] = _stubs["torch"].nn

for _pkg in ("sentence_transformers", "transformers"):
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.SentenceTransformer = MagicMock
        sys.modules[_pkg] = _stub
