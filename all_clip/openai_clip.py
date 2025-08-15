"""https://github.com/openai/CLIP by https://github.com/rom1504/clip for pypi packaging"""

import clip
import torch
import warnings


def _parse_version(version_string):
    """Simple version parsing for major.minor.patch format"""
    try:
        parts = version_string.split(".")
        return tuple(int(part.split("+")[0]) for part in parts[:3])  # Handle versions like "2.8.0+cu121"
    except (ValueError, IndexError):
        return (0, 0, 0)  # Fallback for unparseable versions


def load_openai_clip(clip_model, use_jit, device, clip_cache_path):
    """Load OpenAI CLIP model with PyTorch 2.8+ compatibility fixes."""
    # PyTorch 2.8+ compatibility: automatically disable JIT for OpenAI CLIP models
    # to avoid TorchScript NotImplementedError issues
    pytorch_version = _parse_version(torch.__version__)
    if pytorch_version >= (2, 8, 0) and use_jit:
        warnings.warn(
            f"PyTorch {torch.__version__} detected. Disabling JIT compilation (use_jit=False) "
            "for OpenAI CLIP models to avoid TorchScript compatibility issues. "
            "To suppress this warning, explicitly set use_jit=False.",
            UserWarning,
            stacklevel=2,
        )
        use_jit = False

    # Temporarily patch torch.load to handle weights_only parameter for CLIP model loading
    original_torch_load = torch.load

    def _patched_load(*args, **kwargs):
        # Force weights_only=False for CLIP model compatibility with TorchScript archives
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    try:
        # Apply the patch only during CLIP model loading
        torch.load = _patched_load
        model, preprocess = clip.load(clip_model, device=device, jit=use_jit, download_root=clip_cache_path)
    finally:
        # Always restore the original torch.load
        torch.load = original_torch_load

    def tokenizer(t):
        return clip.tokenize(t, truncate=True)

    return model, preprocess, tokenizer
