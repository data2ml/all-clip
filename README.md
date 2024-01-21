# all_clip
[![pypi](https://img.shields.io/pypi/v/all_clip.svg)](https://pypi.python.org/pypi/all_clip)

Load any clip model with a standardized interface

## Install

pip install all_clip

## Python examples

```python
from all_clip import load_clip
import torch
from PIL import Image
import pathlib


model, preprocess, tokenizer = load_clip("open_clip:ViT-B-32/laion2b_s34b_b79k", device="cpu", use_jit=False)


image = preprocess(Image.open(str(pathlib.Path(__file__).parent.resolve()) + "/CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]
```

Checkout these examples to call this as a lib:
* [example.py](examples/example.py)

## API

This module exposes a single function `load_clip`:

* **clip_model** CLIP model to load (default *ViT-B/32*). See below supported models section.
* **use_jit** uses jit for the clip model (default *True*)
* **warmup_batch_size** warmup batch size (default *1*)
* **clip_cache_path** cache path for clip (default *None*)
* **device** device (default *None*)

## Related projects

* [clip-retrieval](https://github.com/rom1504/clip-retrieval) to use clip for inference, and retrieval
* [open_clip](https://github.com/mlfoundations/open_clip) to train clip models
* [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) to evaluate clip models

## Supported models

### OpenAI

Specify the model as "ViT-B-32"

### Openclip

`"open_clip:ViT-B-32/laion2b_s34b_b79k"` to use the [open_clip](https://github.com/mlfoundations/open_clip)

### HF CLIP

`"hf_clip:patrickjohncyh/fashion-clip"` to use the [hugging face](https://huggingface.co/docs/transformers/model_doc/clip)

### Deepsparse backend

[DeepSparse](https://github.com/neuralmagic/deepsparse) is an inference runtime for fast sparse model inference on CPUs. There is a backend available within clip-retrieval by installing it with `pip install deepsparse-nightly[clip]`, and specifying a `clip_model` with a prepended `"nm:"`, such as [`"nm:neuralmagic/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K-quant-ds"`](https://huggingface.co/neuralmagic/CLIP-ViT-B-32-256x256-DataComp-s34B-b86K-quant-ds) or [`"nm:mgoin/CLIP-ViT-B-32-laion2b_s34b_b79k-ds"`](https://huggingface.co/mgoin/CLIP-ViT-B-32-laion2b_s34b_b79k-ds).

### Japanese clip

[japanese-clip](https://github.com/rinnakk/japanese-clip) provides some models for japanese.
For example one is `ja_clip:rinna/japanese-clip-vit-b-16`

## How to add a model type

Please follow these steps:
1. Add a file to load model in `all_clip/`
2. Define a loading function, that returns a tuple (model, transform, tokenizer). Please see `all_clip/open_clip.py` as an example. 
3. Add the function into `TYPE2FUNC` in `all_clip/main.py`
4. Add the model type in `test_main.py` and `ci.yml`

Remarks:
- The new tokenizer/model must enable to do the following things as https://github.com/openai/CLIP#usage
  - `tokenizer(texts).to(device)`  ... `texts` is a list of string
  - `model.encode_text(tokenized_texts)` ... `tokenized_texts` is a output from `tokenizer(texts).to(device)`
  - `model.encode_image(images)` ... `images` is a image tensor by the `transform`

## For development

Setup a virtualenv:

```
python3 -m venv .env
source .env/bin/activate
pip install -e .
```

to run tests:
```
pip install -r requirements-test.txt
```
then 
```
make lint
make test
```

You can use `make black` to reformat the code

`python -m pytest -x -s -v tests -k "ja_clip"` to run a specific test
