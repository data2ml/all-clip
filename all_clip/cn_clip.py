"""https://github.com/OFA-Sys/Chinese-CLIP"""
import os.path
from typing import Dict

import cn_clip.clip
import torch
from torch import nn


class CnCLIPForBenchmark(nn.Module):
    """
    enable to do model.encode_text(dict_tensor)
    """

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = torch.device(device=device)

    def encode_image(self, image):
        return self.model.encode_image(image)

    def encode_text(self, text):
        return self.model.encode_text(text)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_chinese_clip(clip_model, use_jit, device, clip_cache_path):  # pylint: disable=unused-argument
    """load chinese clip"""
    try:
        from cn_clip.clip.utils import create_model, image_transform, _MODEL_INFO # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "Install `Chinese-CLIP` by `pip install git+https://github.com/OFA-Sys/Chinese-CLIP.git`"
        ) from exc
    cache_dir = clip_cache_path
    model_info = clip_model.split('/')

    clip_model_parts = clip_model.split("/")
    model_name = clip_model_parts[0]
    checkpoint_file = "/".join(clip_model_parts[1:])

    model_name = _MODEL_INFO[model_name]['struct']
    checkpoint = None
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, 'rb') as opened_file:
            # loading saved checkpoint
            checkpoint = torch.load(opened_file, map_location="cpu")
    model = create_model(model_name, checkpoint)
    model.to(device=device, dtype=torch.float32)
    processor = image_transform()

    return CnCLIPForBenchmark(model, device), processor, cn_clip.clip.tokenize
