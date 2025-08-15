"""https://github.com/neuralmagic/deepsparse/tree/bc4ffd305ac52718cc755430a1c44eb48739cfb4/src/deepsparse/clip"""

from torch import nn
import torch

import numpy as np
import clip


class DeepSparseWrapper(nn.Module):
    """
    Wrap DeepSparse for managing input types
    """

    def __init__(self, model_path):
        super().__init__()

        import deepsparse  # pylint: disable=import-outside-toplevel

        ##### Fix for two-input models
        from deepsparse.clip import CLIPTextPipeline  # pylint: disable=import-outside-toplevel

        def custom_process_inputs(self, inputs):
            if not isinstance(inputs.text, list):
                # Always wrap in a list
                inputs.text = [inputs.text]
            if not isinstance(inputs.text[0], str):
                # If not a string, assume it's already been tokenized
                tokens = np.stack(inputs.text, axis=0, dtype=np.int32)
                return [tokens, np.array(tokens.shape[0] * [tokens.shape[1] - 1])]
            else:
                tokens = [np.array(t).astype(np.int32) for t in self.tokenizer(inputs.text)]
                tokens = np.stack(tokens, axis=0)
                return [tokens, np.array(tokens.shape[0] * [tokens.shape[1] - 1])]

        # This overrides the process_inputs function globally for all CLIPTextPipeline classes
        CLIPTextPipeline.process_inputs = custom_process_inputs
        ####

        self.textual_model_path = model_path + "/textual.onnx"
        self.visual_model_path = model_path + "/visual.onnx"

        self.textual_model = deepsparse.Pipeline.create(task="clip_text", model_path=self.textual_model_path)
        self.visual_model = deepsparse.Pipeline.create(task="clip_visual", model_path=self.visual_model_path)

    def encode_image(self, image):
        image = [np.array(image)]
        embeddings = self.visual_model(images=image).image_embeddings[0]
        return torch.from_numpy(embeddings)

    def encode_text(self, text):
        text = [t.numpy() for t in text]
        embeddings = self.textual_model(text=text).text_embeddings[0]
        return torch.from_numpy(embeddings)

    def forward(self, *args, **kwargs):  # pylint: disable=unused-argument
        return NotImplemented


def load_deepsparse(clip_model, use_jit, device, clip_cache_path):  # pylint: disable=unused-argument
    """load deepsparse"""

    from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel

    # Download the model from HF
    model_folder = snapshot_download(repo_id=clip_model)
    # Compile the model with DeepSparse
    model = DeepSparseWrapper(model_path=model_folder)

    from deepsparse.clip.constants import CLIP_RGB_MEANS, CLIP_RGB_STDS  # pylint: disable=import-outside-toplevel

    def process_image(image):
        image = model.visual_model._preprocess_transforms(image.convert("RGB"))  # pylint: disable=protected-access
        image_array = np.array(image)
        image_array = image_array.transpose(2, 0, 1).astype("float32")
        image_array /= 255.0
        image_array = (image_array - np.array(CLIP_RGB_MEANS).reshape((3, 1, 1))) / np.array(CLIP_RGB_STDS).reshape(
            (3, 1, 1)
        )
        return torch.from_numpy(np.ascontiguousarray(image_array, dtype=np.float32))

    def tokenizer(t):
        return clip.tokenize(t, truncate=True)

    return model, process_image, tokenizer
