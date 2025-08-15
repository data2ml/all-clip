"""load clip"""

from functools import lru_cache
import torch
from PIL import Image
import time

from .deepsparse_clip import load_deepsparse
from .hf_clip import load_hf_clip
from .open_clip import load_open_clip
from .openai_clip import load_openai_clip
from .ja_clip import load_japanese_clip


_CLIP_REGISTRY = {
    "open_clip:": load_open_clip,
    "hf_clip:": load_hf_clip,
    "nm:": load_deepsparse,
    "ja_clip:": load_japanese_clip,
    "openai_clip:": load_openai_clip,
    "": load_openai_clip,
}


@lru_cache(maxsize=None)
def load_clip_without_warmup(clip_model, use_jit, device, clip_cache_path):
    """Load clip"""

    for prefix, loader in _CLIP_REGISTRY.items():
        if clip_model.startswith(prefix):
            clip_model = clip_model[len(prefix) :]
            model, preprocess, tokenizer = loader(clip_model, use_jit, device, clip_cache_path)
            return model, preprocess, tokenizer

    raise ValueError(f"Unknown clip model {clip_model}")


@lru_cache(maxsize=None)
def load_clip(
    clip_model="ViT-B/32",
    use_jit=True,
    warmup_batch_size=1,
    clip_cache_path=None,
    device=None,
):
    """Load clip then warmup"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_clip_without_warmup(clip_model, use_jit, device, clip_cache_path)

    start = time.time()
    print(f"warming up with batch size {warmup_batch_size} on {device}", flush=True)
    warmup(warmup_batch_size, device, preprocess, model, tokenizer)
    duration = time.time() - start
    print(f"done warming up in {duration}s", flush=True)
    return model, preprocess, tokenizer


def warmup(batch_size, device, preprocess, model, tokenizer):
    fake_img = Image.new("RGB", (224, 224), color="red")
    fake_text = ["fake"] * batch_size
    image_tensor = torch.cat([torch.unsqueeze(preprocess(fake_img), 0)] * batch_size).to(device)
    text_tokens = tokenizer(fake_text).to(device)
    for _ in range(2):
        with torch.no_grad():
            model.encode_image(image_tensor)
            model.encode_text(text_tokens)
