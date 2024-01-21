"""https://github.com/rinnakk/japanese-clip"""

from typing import Dict
import torch
from torch import nn


class DictTensor(dict):
    """
    enable to do `tokenizer(texts).to(device)`
    """

    def __init__(self, d: Dict[str, torch.Tensor]):
        super().__init__()
        self.d = d

    def __getitem__(self, key):
        return self.d[key]

    def __setitem__(self, key, value):
        self.d[key] = value

    def __iter__(self):
        return iter(self.d)

    def keys(self):
        return self.d.keys()

    def __repr__(self):
        return repr(self.d)

    def to(self, device):
        return {k: v.to(device) for k, v in self.d.items()}


class JaCLIPForBenchmark(nn.Module):
    """
    enable to do model.encode_text(dict_tensor)
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode_text(self, dict_tensor):
        return self.model.get_text_features(**dict_tensor)

    def encode_image(self, image):
        return self.model.get_image_features(image)

    def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
        return NotImplemented


def load_japanese_clip(clip_model, use_jit, device, clip_cache_path):  # pylint: disable=unused-argument
    """
    Load Japanese CLIP/CLOOB by rinna (https://github.com/rinnakk/japanese-clip)
    Remarks:
     - You must input not only input_ids but also attention_masks and
     position_ids when doing `model.encode_text()` to make it work correctly.
    """
    try:
        import japanese_clip as ja_clip  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ImportError(
            "Install `japanese_clip` by `pip install git+https://github.com/rinnakk/japanese-clip.git`"
        ) from exc
    cache_dir = clip_cache_path
    model, transform = ja_clip.load(clip_model, device=device, cache_dir=cache_dir)

    class JaTokenizerForBenchmark:
        """Tokenizer for japanese-clip"""

        def __init__(
            self,
        ):
            self.tokenizer = ja_clip.load_tokenizer()

        def __call__(self, texts) -> Dict[str, torch.Tensor]:
            inputs = ja_clip.tokenize(texts, tokenizer=self.tokenizer, device="cpu")
            return DictTensor(inputs)

        def __len__(self):
            return len(self.tokenizer)

    return JaCLIPForBenchmark(model), transform, JaTokenizerForBenchmark()
