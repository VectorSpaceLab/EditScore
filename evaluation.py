import dotenv

dotenv.load_dotenv(override=True)

import argparse
import glob
import hashlib
import json
import logging
import os
import time
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional

import dotenv
from PIL import Image
from tqdm import tqdm
from datasets import Dataset, load_dataset

from editscore import EditScore

PROMPT_FOLLOWING = "prompt_following"
CONSISTENCY = "consistency"
OVERALL = "overall"
SCORE_CATEGORIES = [PROMPT_FOLLOWING, CONSISTENCY, OVERALL]


class CacheManager:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.lock = threading.Lock()
        self.cache = self._load()

    def _load(self) -> Dict[str, Any]:
        cache = {}
        if not os.path.exists(self.cache_file):
            print(
                f"Cache file not found at {self.cache_file}. A new one will be created."
            )
            return cache

        with open(self.cache_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    cache[data["key"]] = data["result"]
                except json.JSONDecodeError:
                    logging.warning(
                        f"Skipping corrupted line {i + 1} in cache file: {line.strip()}"
                    )
        print(f"Loaded {len(cache)} items from {self.cache_file}.")
        return cache

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def append(self, key: str, result: Any):
        with self.lock:
            self.cache[key] = result
            with open(self.cache_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"key": key, "result": result}, ensure_ascii=False)
                    + "\n"
                )

def generate_cache_key(pair_key):
    return hashlib.sha256(pair_key.encode("utf-8")).hexdigest()

def load_pairs_dataset(dataset: Dataset) -> Dict[str, Tuple[str, Image.Image, Image.Image]]:
    pairs = {}
    for data in dataset:
        key1, key2 = data["key"]
        instruction = data["instruction"]
        input_image = data["input_image"].convert("RGB")

        pairs[key1] = (instruction, input_image, data["output_images"][0].convert("RGB"))
        pairs[key2] = (instruction, input_image, data["output_images"][1].convert("RGB"))
    return pairs

def _load_item(data: dict) -> list[tuple[str, tuple[str, Image.Image, Image.Image]]]:
    key1, key2 = data["key"]
    instruction = data["instruction"]
    
    input_image = data["input_image"].convert("RGB")
    output_image1 = data["output_images"][0].convert("RGB")
    output_image2 = data["output_images"][1].convert("RGB")

    return [
        (key1, (instruction, input_image, output_image1)),
        (key2, (instruction, input_image, output_image2)),
    ]

def load_pairs_dataset_multithreaded(dataset: Dataset, max_workers: int = None) -> Dict[str, Tuple[str, Image.Image, Image.Image]]:
    if max_workers is None:
        # max_workers = min(32, (os.cpu_count() or 1) * 5)  
        max_workers = os.cpu_count() or 1

    pairs = {}
    
    print(f"Processing dataset (length: {len(dataset)}) with {max_workers} threads", flush=True)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = tqdm(
            executor.map(_load_item, dataset), 
            total=len(dataset), 
            desc="Processing dataset with multiple threads"
        )
        
        for result_pairs in results_iterator:
            pairs.update(result_pairs)
            
    return pairs


def process_single_item(key, item, scorer):
    instruction = item[0]
    input_image = item[1]
    output_image = item[2]
    
    output_image = output_image.resize((input_image.size[0], input_image.size[1]))

    score = scorer.evaluate([input_image, output_image], instruction)
    return key, score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark_dir", type=str, default="EditScore/EditReward-Bench"
    )
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument(
        "--backbone",
        type=str,
        default="openai",
        choices=["openai", "qwen25vl", "qwen25vl_vllm", "internvl3_5"],
    )
    parser.add_argument("--model_name_or_path", type=str, default="gpt-4.1")
    parser.add_argument(
        "--openai_url", type=str, default="https://api.openai.com/v1/chat/completions"
    )
    parser.add_argument("--key", type=str, default="PUT YOUR API KEY HERE")
    parser.add_argument("--num_pass", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--score_range", type=int, default=25)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=1536)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--max_num_batched_tokens", type=int, default=1536)
    parser.add_argument("--enable_lora", action="store_true")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default=None)
    return parser.parse_args()


def main(args):
    start_time = time.time()
    scorer = EditScore(
        backbone=args.backbone,
        key=args.key,
        openai_url=args.openai_url,
        model_name_or_path=args.model_name_or_path,
        score_range=args.score_range,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        num_pass=args.num_pass,
        enable_lora=args.enable_lora,
        lora_path=args.lora_path,
        cache_dir=args.cache_dir,
    )
    print(f"Scorer initialized in {time.time() - start_time} seconds", flush=True)

    cache_dir = os.path.join(args.result_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f"{args.backbone}_{args.model_name_or_path.replace('/', '_')}.jsonl"
    )
    cache_manager = CacheManager(cache_file)

    start_time = time.time()
    dataset = load_dataset(args.benchmark_dir, split="train")
    print(f"Dataset loaded in {time.time() - start_time} seconds", flush=True)

    start_time = time.time()
    unique_pairs = load_pairs_dataset_multithreaded(dataset)
    print(f"Pairs loaded in {time.time() - start_time} seconds", flush=True)

    all_scores = {}
    pairs_to_process = [
        pair_key
        for pair_key in unique_pairs.keys()
        if cache_manager.get(generate_cache_key(pair_key)) is None
    ]

    for pair_key in unique_pairs.keys():
        if pair_key not in pairs_to_process:
            all_scores[pair_key] = cache_manager.get(generate_cache_key(pair_key))

    print(
        f"{len(unique_pairs) - len(pairs_to_process)} pairs found in cache. Processing {len(pairs_to_process)} new pairs.",
        flush=True
    )

    if pairs_to_process:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(process_single_item, pair_key, unique_pairs[pair_key], scorer)
                for pair_key in pairs_to_process
            ]

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                unit="pair",
                desc="Processing",
            ):
                pair_key, result = future.result()
                if result:
                    all_scores[pair_key] = result
                    cache_manager.append(generate_cache_key(pair_key), result)

    print("Writing results...", flush=True)

    start_time = time.time()
    # dataset = dataset.remove_columns(["input_image", "output_images"])
    for idx, data in enumerate(dataset):
        key1, key2 = data["key"]
        task_type = data["task_type"]
        dimension = data["dimension"]

        score1 = all_scores[key1][dimension]
        score2 = all_scores[key2][dimension]
        data["score"] = [score1, score2]
        
        input_image_path = os.path.join(args.result_dir, "images", f"{key1}_input.png")
        output_image_path1 = os.path.join(args.result_dir, "images", f"{key1}.png")
        output_image_path2 = os.path.join(args.result_dir, "images", f"{key2}.png")

        os.makedirs(os.path.dirname(input_image_path), exist_ok=True)

        data['input_image'].save(input_image_path)
        data['output_images'][0].save(output_image_path1)
        data['output_images'][1].save(output_image_path2)

        json_line = {
            "key": (key1, key2),
            "idx": idx,
            "score": [score1, score2],
            "SC_reasoning": [all_scores[key1]["SC_reasoning"], all_scores[key2]["SC_reasoning"]],
            "PQ_reasoning": [all_scores[key1]["PQ_reasoning"], all_scores[key2]["PQ_reasoning"]],
            "input_image": input_image_path,
            "output_images": [output_image_path1, output_image_path2],
        }

        save_file = os.path.join(
            args.result_dir, args.backbone, task_type, f"{dimension}.jsonl"
        )
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

        with open(save_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

    print(f"Results written in {time.time() - start_time} seconds", flush=True)
    print("--- Completed! ---", flush=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)
