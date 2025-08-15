"""https://huggingface.co/docs/transformers/model_doc/clip"""

import torch
from torch import autocast, nn
import clip


class HFClipWrapper(nn.Module):
    """
    Wrap Huggingface ClipModel
    """

    def __init__(self, inner_model, device):
        super().__init__()
        self.inner_model = inner_model
        self.device = torch.device(device=device)
        if self.device.type == "cpu":
            self.dtype = torch.float32
        else:
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def encode_image(self, image):
        if self.device.type == "cpu":
            return self.inner_model.get_image_features(image.squeeze(1))
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.get_image_features(image.squeeze(1))

    def encode_text(self, text):
        if self.device.type == "cpu":
            return self.inner_model.get_text_features(text)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.get_text_features(text)

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)


def load_hf_clip(clip_model, use_jit, device, clip_cache_path):  # pylint: disable=unused-argument
    """load hf clip"""
    from transformers import CLIPProcessor, CLIPModel  # pylint: disable=import-outside-toplevel

    model = CLIPModel.from_pretrained(clip_model)
    preprocess = CLIPProcessor.from_pretrained(clip_model).image_processor
    model = HFClipWrapper(inner_model=model, device=device)
    model.to(device=device)

    def tokenizer(t):
        return clip.tokenize(t, truncate=True)

    return model, lambda x: preprocess(x, return_tensors="pt").pixel_values, tokenizer
