""" https://github.com/openai/CLIP by https://github.com/rom1504/clip for pypi packaging """

import clip


def load_openai_clip(clip_model, use_jit, device, clip_cache_path):
    model, preprocess = clip.load(clip_model, device=device, jit=use_jit, download_root=clip_cache_path)

    def tokenizer(t):
        return clip.tokenize(t, truncate=True)

    return model, preprocess, tokenizer
