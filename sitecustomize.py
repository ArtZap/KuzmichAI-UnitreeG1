"""Runtime compatibility fixes for Jetson Python 3.8.

NVIDIA's CUDA-enabled PyTorch wheel for this JetPack is currently older than
the PyTorch version expected by Coqui TTS 0.22.  Coqui imports
``torch.nn.utils.parametrizations.weight_norm`` during package initialization;
PyTorch 2.0 keeps the callable at ``torch.nn.utils.weight_norm``.  Expose the
same name before Coqui imports so XTTS can run with GPU acceleration.
"""

try:
    import torch.nn.utils as _utils
    import torch.nn.utils.parametrizations as _parametrizations

    if not hasattr(_parametrizations, "weight_norm") and hasattr(_utils, "weight_norm"):
        _parametrizations.weight_norm = _utils.weight_norm
except Exception:
    pass
