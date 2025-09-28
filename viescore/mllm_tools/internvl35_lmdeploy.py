from typing import List

import random
# import magic
# import megfile

import numpy as np
import torch

from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN


def set_seed(seed: int):
    """
    Args:
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_chat_template(prompt, num_images: int = 2):
    """
    This is used since the bug of transformers which do not support vision id https://github.com/QwenLM/Qwen2.5-VL/issues/716#issuecomment-2723316100
    """
    template = "\n".join([f"Image-{i}: {IMAGE_TOKEN}" for i in range(1, num_images + 1)])
    template += f"\n{prompt}"
    return template

class InternVL35():
    def __init__(self, model, max_model_len: int = 16384, tensor_parallel_size=1, max_num_seqs=32) -> None:
        # attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else None
        self.model = pipeline(model, backend_config=PytorchEngineConfig(session_len=max_model_len, tp=tensor_parallel_size))
        
    def prepare_input(self, images: List = [], text_prompt: str = ""):
        if not isinstance(images, list):
            images = [images]
        messages = (apply_chat_template(text_prompt, num_images=len(images)), images)
        return messages

    def inference(self, messages):
        set_seed(42)
        # Prepare the inputs

        response = self.model(messages)
        print(f"{response.text=}", flush=True)
        return response.text

if __name__ == "__main__":
    model = InternVL35(
        vlm_model="OpenGVLab/InternVL3_5-8B",
        max_model_len=16384,
        tensor_parallel_size=1,
        max_num_seqs=32
    )

    from PIL import Image
    prompt = model.prepare_input(
        [Image.open("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")], 
        'Describe the image in detail.'
    )

    prompt2 = model.prepare_input(
        [Image.open("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")], 
        'How well it looks? Give a score between 0 and 100.'
    )
    res = model.inference([prompt, prompt2])
    print("result : \n", res)