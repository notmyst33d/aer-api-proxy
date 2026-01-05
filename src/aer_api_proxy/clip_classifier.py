# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Myst33d <myst33d@gmail.com>

import numpy
import onnxruntime
from PIL import Image
from tokenizers import Tokenizer
from typing import List, Tuple

IMAGE_MEAN = numpy.array([0.48145466, 0.4578275, 0.40821073])
IMAGE_STD = numpy.array([0.26862954, 0.26130258, 0.27577711])


def softmax(x: numpy.ndarray, dim: int = 0) -> numpy.ndarray:
    x_max = numpy.max(x, axis=dim, keepdims=True)
    e_x = numpy.exp(x - x_max)
    return e_x / numpy.sum(e_x, axis=dim, keepdims=True)


def image_to_tensor(im: Image.Image) -> numpy.ndarray:
    im = im.convert("RGB")
    im = im.resize((224, 224))
    im_arr = numpy.array(im, dtype=numpy.float64)
    im_arr = im_arr / 255.0
    im_arr = (im_arr - IMAGE_MEAN) / IMAGE_STD
    return im_arr.transpose((2, 0, 1)).astype(numpy.float32)


def labels_to_tensor(
    tokenizer: Tokenizer,
    labels: List[str],
) -> Tuple[List[List[int]], List[List[int]]]:
    input_ids = []
    attention_mask = []
    for label in labels:
        encoding = tokenizer.encode(label)
        input_ids.append(encoding.ids)
        attention_mask.append(encoding.attention_mask)

    # Add padding
    max_length = max(map(lambda x: len(x), input_ids))
    pad_id = input_ids[0][-1]
    input_ids = list(map(lambda x: x + [pad_id] * (max_length - len(x)), input_ids))
    attention_mask = list(
        map(lambda x: x + [0] * (max_length - len(x)), attention_mask)
    )

    return (input_ids, attention_mask)


class CLIPClassifier:
    def __init__(
        self,
        model_path: str = "./clip_model.onnx",
        tokenizer_path: str = "./clip_tokenizer.json",
    ):
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self._session = onnxruntime.InferenceSession(
            model_path, sess_options=session_options
        )
        self._tokenizer = Tokenizer.from_file(tokenizer_path)

    def run(self, image: Image.Image, labels: List[str]) -> numpy.ndarray:
        pixel_values = image_to_tensor(image)
        input_ids, attention_mask = labels_to_tensor(self._tokenizer, labels)
        inputs = {
            "pixel_values": [pixel_values],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        outputs = self._session.run(["logits_per_image"], inputs)
        if isinstance(outputs[0], numpy.ndarray):
            return softmax(outputs[0], dim=1)
        else:
            raise Exception("unknown return type")
