import sys
sys.path.insert(0, 'editscore')

from typing import Optional
from .utils import (
    mllm_output_to_dict
)
import math
from . import vie_prompts
import numpy as np
from .json_parser import parse_vlm_output_to_dict

class EditScore:
    def __init__(
        self,
        backbone="gpt-4.1",
        openai_url="https://api.openai.com/v1/chat/completions",
        key=None,
        model_name_or_path="",
        score_range: int=25,
        temperature: float=0.7,
        tensor_parallel_size: int=1,
        max_model_len: int=1536,
        max_num_batched_tokens: int=1536,
        max_num_seqs: int=32,
        num_pass: int=1,
        reduction: str="average_last",
        seed: int=42,
        lora_path: Optional[str]=None,
        cache_dir: Optional[str]=None,
    ) -> None:
        self.backbone = backbone
        self.score_range = score_range
        self.reduction = reduction
        self.seed = seed
        self.num_pass = num_pass

        if self.backbone == 'openai':
            from .mllm_tools.openai import GPT4o
            self.model = GPT4o(key, model_name=model_name_or_path, url=openai_url)
        elif self.backbone == "qwen25vl":
            from .mllm_tools.qwen25vl import Qwen25VL
            self.model = Qwen25VL(
                vlm_model=model_name_or_path,
                temperature=temperature,
                seed=seed,
                lora_path=lora_path,
            )
        elif self.backbone == "qwen25vl_vllm":
            from .mllm_tools.qwen25vl_vllm import Qwen25VL
            self.model = Qwen25VL(
                vlm_model=model_name_or_path,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                temperature=temperature,
                seed=seed,
                lora_path=lora_path,
                cache_dir=cache_dir,
            )
        elif self.backbone == "internvl3_5":
            from .mllm_tools.internvl35_lmdeploy import InternVL35
            self.model = InternVL35(model=model_name_or_path, tensor_parallel_size=tensor_parallel_size)
            
        self.context = vie_prompts._context_no_delimit_reasoning_first
    
        self.SC_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_two_image_edit_rule, vie_prompts._prompts_0shot_tie_rule_SC.replace('10', str(self.score_range))])
        self.PQ_prompt = "\n".join([self.context, vie_prompts._prompts_0shot_rule_PQ.replace('10', str(self.score_range))])

    def evaluate(self, image_prompts, text_prompt):
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]

        if self.backbone in ['openai']:
            self.model.use_encode = False if isinstance(image_prompts[0], str) else True
            
        _SC_prompt = self.SC_prompt.replace("<instruction>", text_prompt)
        
        SC_prompt_final = self.model.prepare_input(image_prompts, _SC_prompt)
        PQ_prompt_final = self.model.prepare_input(image_prompts[-1], self.PQ_prompt) # assume the last image is the edited image

        outputs_multi_pass = []

        for i in range(self.num_pass):
            SC_dict = False
            PQ_dict = False
            tries = 0
            max_tries = 2
            while SC_dict is False or PQ_dict is False:
                tries += 1
                give_up_parsing = True if tries > max_tries else False

                result_SC = self.model.inference(SC_prompt_final, seed=self.seed + i)
                result_PQ = self.model.inference(PQ_prompt_final, seed=self.seed + i)

                if result_SC in ["I'm sorry, but I can't assist with that request."] or result_PQ in ["I'm sorry, but I can't assist with that request."]:
                    give_up_parsing = True
                    
                SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=give_up_parsing, text_prompt=text_prompt, score_range=self.score_range)
                PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=give_up_parsing, text_prompt=text_prompt, score_range=self.score_range)

            if SC_dict == "rate_limit_exceeded" or PQ_dict == "rate_limit_exceeded":
                print("rate_limit_exceeded") 
                raise ValueError("rate_limit_exceeded")
            
            try:
                SC_score = min(SC_dict['score']) / (self.score_range / 10)
                PQ_score = min(PQ_dict['score']) / (self.score_range / 10)
                O_score = math.sqrt(SC_score * PQ_score)
            except Exception as e:
                print(f"{e=} {SC_dict['score']=} {PQ_dict['score']=}")
                raise e

            try:
                outputs_multi_pass.append({
                    'prompt_following': SC_dict['score'][0] / (self.score_range / 10),
                    'consistency': SC_dict['score'][1] / (self.score_range / 10),
                    'perceptual_quality': PQ_score,
                    'overall': O_score,
                })
            except Exception as e:
                print(f"{e=} {SC_dict['score']=} {PQ_dict['score']=}")
                raise e
        
        output = {
                    "prompt_following": np.mean([output_per_pass["prompt_following"] for output_per_pass in outputs_multi_pass]),
                    "consistency": np.mean([output_per_pass["consistency"] for output_per_pass in outputs_multi_pass]),
                    "perceptual_quality": np.mean([output_per_pass["perceptual_quality"] for output_per_pass in outputs_multi_pass]),
                    "overall": np.mean([output_per_pass["overall"] for output_per_pass in outputs_multi_pass]),
                    "SC_reasoning": SC_dict["reasoning"],
                    "PQ_reasoning": PQ_dict["reasoning"],
                }
        if self.reduction == "average_first":
            output["overall"] = math.sqrt(output["prompt_following"] * output["perceptual_quality"])
        return output


    def batch_evaluate(self, image_prompts, text_prompt):
        SC_prompt = [self.SC_prompt.replace("<instruction>", _text_prompt) for _text_prompt in text_prompt]

        SC_prompt = [self.model.prepare_input(image_prompt, _SC_prompt) for image_prompt, _SC_prompt in zip(image_prompts, SC_prompt)]
        PQ_prompt = [self.model.prepare_input(image_prompt, self.PQ_prompt) for image_prompt in image_prompts]

        outputs_multi_pass = [[] for _ in range(len(image_prompts))]
        for i in range(self.num_pass):
            results = self.model.batch_inference(SC_prompt + PQ_prompt, seed=self.seed + i)

            SC_evaluations = [parse_vlm_output_to_dict(results[i]) for i in range(len(results) // 2)]
            PQ_evaluations = [parse_vlm_output_to_dict(results[i]) for i in range(len(results) // 2, len(results))]

            for idx, (SC_evaluation, PQ_evaluation) in enumerate(zip(SC_evaluations, PQ_evaluations)):
                SC_scores = SC_evaluation["score"]
                PQ_scores = PQ_evaluation["score"]

                if len(SC_scores) == 0:
                    SC_scores = [self.score_range / 2]
                if len(PQ_scores) == 0:
                    PQ_scores = [self.score_range / 2]

                SC_score = min(SC_scores) / (self.score_range / 10)
                PQ_score = min(PQ_scores) / (self.score_range / 10)
                if SC_score < 0 or SC_score > 10:
                    SC_score = self.score_range / 2
                if PQ_score < 0 or PQ_score > 10:
                    PQ_score = self.score_range / 2
                O_score = math.sqrt(SC_score * PQ_score)

                outputs_multi_pass[idx].append(
                    {
                        "SC_score": SC_score,
                        "PQ_score": PQ_score,
                        "O_score": O_score,
                        "SC_score_reasoning": SC_evaluation["reasoning"],
                        "PQ_score_reasoning": PQ_evaluation["reasoning"],
                        "SC_raw_output": results[idx],
                        "PQ_raw_output": results[len(results) // 2 + idx],
                    }
                )
        
        outputs = []
        for idx, outputs_per_prompt in enumerate(outputs_multi_pass):
            outputs.append(
                {
                    "SC_score": np.mean([output_per_pass["SC_score"] for output_per_pass in outputs_per_prompt]),
                    "PQ_score": np.mean([output_per_pass["PQ_score"] for output_per_pass in outputs_per_prompt]),
                    "O_score": np.mean([output_per_pass["O_score"] for output_per_pass in outputs_per_prompt]),
                    "SC_score_reasoning": outputs_per_prompt[0]["SC_score_reasoning"],
                    "PQ_score_reasoning": outputs_per_prompt[0]["PQ_score_reasoning"],
                    "SC_raw_output": outputs_per_prompt[0]["SC_raw_output"],
                    "PQ_raw_output": outputs_per_prompt[0]["PQ_raw_output"],
                }
            )
            if self.reduction == "average_first":
                outputs[-1]["O_score"] = math.sqrt(outputs[-1]["SC_score"] * outputs[-1]["PQ_score"])
        return outputs