from typing import Optional

import os
import random
import time
import numpy as np
import torch

from vllm import LLM
from vllm.sampling_params import SamplingParams

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

from qwen_vl_utils import process_vision_info


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
    template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    template += "".join([f"<img{i}>: <|vision_start|><|image_pad|><|vision_end|>" for i in range(1, num_images + 1)])
    template += f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return template


class Qwen25VL():
    def __init__(
        self,
        vlm_model,
        max_model_len: int = 1536,
        tensor_parallel_size=1,
        max_num_seqs=32,
        max_num_batched_tokens=1536,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        enable_lora: bool = False,
        lora_path: str = "",
        cache_dir: Optional[str] = None,
    ) -> None:
        self.enable_lora = enable_lora
        self.lora_path = lora_path

        if self.enable_lora:
            if cache_dir is None:
                root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                cache_dir = os.path.join(root_dir, "cache", f"{os.path.basename(vlm_model)}_merged_lora")

            if not os.path.exists(cache_dir):
                print(f"Merging LORA to {vlm_model} and saving to {cache_dir}", flush=True)
                start_time = time.time()
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    vlm_model, torch_dtype=torch.bfloat16, device_map="cpu"
                )
                model = PeftModel.from_pretrained(model, lora_path)
                model = model.merge_and_unload()
                model.save_pretrained(cache_dir)

                processor = AutoProcessor.from_pretrained(vlm_model)
                processor.save_pretrained(cache_dir)

                print(f"Merging LORA to {vlm_model} and saving to {cache_dir} took {time.time() - start_time} seconds", flush=True)
            else:
                print(f"Skipping merging LORA, as merged model already exists in {cache_dir}", flush=True)

            vlm_model = cache_dir

        self.model = LLM(
            model=vlm_model,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            limit_mm_per_prompt={"image": 2},
            enable_prefix_caching=True,
        )
        self.temperature = temperature
        self.seed = seed
    
    def prepare_input(self, images, text_prompt: str = ""):
        if not isinstance(images, list):
            images = [images]

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": text_prompt}],
            }
        ]
        text = apply_chat_template(text_prompt, num_images=len(images))
        image_inputs, _ = process_vision_info(messages)

        messages = {
            "prompt": text,
            "multi_modal_data": {"image": image_inputs},
        }
        return messages

    def inference(self, messages, seed: Optional[int] = None):
        seed = self.seed if seed is None else seed
        sampling_params = SamplingParams(max_tokens=512, temperature=self.temperature, top_p=0.9, top_k=20, seed=seed)
        outputs = self.model.generate(messages, sampling_params, use_tqdm=False)

        responses = []
        for output in outputs:
            instruction = output.outputs[0].text.strip()
            responses.append(instruction)

        return responses[0]