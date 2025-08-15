"""https://github.com/mlfoundations/open_clip"""

from torch import autocast, nn
import torch


class OpenClipWrapper(nn.Module):
    """
    Wrap OpenClip for managing input types
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
            return self.inner_model.encode_image(image)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.encode_image(image)

    def encode_text(self, text):
        if self.device.type == "cpu":
            return self.inner_model.encode_text(text)
        with autocast(device_type=self.device.type, dtype=self.dtype):
            return self.inner_model.encode_text(text)

    def forward(self, *args, **kwargs):
        return self.inner_model(*args, **kwargs)


def load_open_clip(clip_model, use_jit=True, device="cuda", clip_cache_path=None):
    """load open clip"""

    import open_clip  # pylint: disable=import-outside-toplevel

    torch.backends.cuda.matmul.allow_tf32 = True
    clip_model_parts = clip_model.split("/")
    clip_model = clip_model_parts[0]
    checkpoint = "/".join(clip_model_parts[1:])
    if checkpoint == "":
        pretrained = dict(open_clip.list_pretrained())
        checkpoint = pretrained[clip_model]
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model,
        pretrained=checkpoint,
        device=device,
        jit=use_jit,
        cache_dir=clip_cache_path,
    )
    model = OpenClipWrapper(inner_model=model, device=device)
    model.to(device=device)

    return model, preprocess, open_clip.get_tokenizer(clip_model)
