from typing import Optional
import random
import numpy as np
import torch

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel


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
        temperature: float = 0.7,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
    ) -> None:
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_model, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.merge_and_unload()

        self.processor = AutoProcessor.from_pretrained(vlm_model)
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
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return inputs

    def inference(self, inputs, seed: Optional[int] = None):
        seed = self.seed if seed is None else seed

        set_seed(seed)
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9,
            top_k=20,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        outputs = [output.strip() for output in outputs]
        return outputs[0]