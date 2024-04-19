from all_clip import load_clip
import torch
from PIL import Image
import pathlib
import pytest


@pytest.mark.parametrize(
    "model",
    [
        "openai_clip:ViT-B/32",
        "open_clip:ViT-B-32/laion2b_s34b_b79k",
        "hf_clip:patrickjohncyh/fashion-clip",
        "nm:mgoin/CLIP-ViT-B-32-laion2b_s34b_b79k-ds",
        "ja_clip:rinna/japanese-clip-vit-b-16",
        "cn_clip:ViT-B-16/no_checkpoint"
    ],
)
def test_load_clip(model):
    model, preprocess, tokenizer = load_clip(model, device="cpu", use_jit=False)

    image = preprocess(Image.open(str(pathlib.Path(__file__).parent.resolve()) + "/CLIP.png")).unsqueeze(0)
    text = tokenizer(["a diagram", "a dog", "a cat"])

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
