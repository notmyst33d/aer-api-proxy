# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Myst33d <myst33d@gmail.com>

import os
import subprocess
from typing import NamedTuple


class Model(NamedTuple):
    path: str
    url: str


CLASSIFIER_MODEL = Model(
    "classifier_model.onnx",
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/model_q4.onnx",
)
CLASSIFIER_TOKENIZER = Model(
    "classifier_tokenizer.json",
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/tokenizer.json",
)
LLM_MODEL = Model(
    "llm_model.gguf",
    "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/Qwen3VL-2B-Instruct-Q4_K_M.gguf",
)
LLM_MODEL_MMPROJ = Model(
    "llm_model_mmproj.gguf",
    "https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
)


def ensure_models():
    if (
        os.path.isfile(CLASSIFIER_MODEL.path)
        and os.path.isfile(CLASSIFIER_TOKENIZER.path)
        and os.path.isfile(LLM_MODEL.path)
        and os.path.isfile(LLM_MODEL_MMPROJ.path)
    ):
        return

    for file in [
        CLASSIFIER_MODEL,
        CLASSIFIER_TOKENIZER,
        LLM_MODEL,
        LLM_MODEL_MMPROJ,
    ]:
        if os.path.isfile(file.path):
            continue
        subprocess.call(["curl", "-Lo", file.path, file.url])
