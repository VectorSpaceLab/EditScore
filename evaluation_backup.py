import dotenv

dotenv.load_dotenv(override=True)

import argparse
import glob
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Optional

import dotenv
from PIL import Image
from tqdm import tqdm

from viescore import VIEScore

PROMPT_FOLLOWING = "prompt_following"
CONSISTENCY = "consistency"
OVERALL = "overall"
SCORE_CATEGORIES = [PROMPT_FOLLOWING, CONSISTENCY, OVERALL]

Pair = Tuple[str, str, str]

class CacheManager:
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.lock = threading.Lock()
        self.cache = self._load()

    def _load(self) -> Dict[str, Any]:
        cache = {}
        if not os.path.exists(self.cache_file):
            logging.info(f"Cache file not found at {self.cache_file}. A new one will be created.")
            return cache
        
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    cache[data['key']] = data['result']
                except json.JSONDecodeError:
                    logging.warning(f"Skipping corrupted line {i+1} in cache file: {line.strip()}")
        logging.info(f"Loaded {len(cache)} items from {self.cache_file}.")
        return cache

    def get(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def append(self, key: str, result: Any):
        with self.lock:
            self.cache[key] = result
            with open(self.cache_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'key': key, 'result': result}, ensure_ascii=False) + '\n')

def generate_cache_key(pair):
    instruction, input_image, output_image = pair
    key_string = f"{instruction}|||{input_image}|||{output_image}"
    return hashlib.sha256(key_string.encode('utf-8')).hexdigest()

def load_pairs_from_paths(jsonl_paths: List[str]) -> set[Pair]:
    pairs = set()
    for file_path in jsonl_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'instruction' in data and 'input_images' in data and 'output_images' in data:
                            instruction = data['instruction']
                            input_image = data['input_images'][0]
                            for output_image in data['output_images']:
                                pairs.add((instruction, input_image, output_image))
                        else:
                            logging.warning(f"Skipping line due to missing keys in {file_path}: {line.strip()}")
                    except (json.JSONDecodeError, IndexError) as e:
                        logging.warning(f"Skipping malformed line in {file_path}: {line.strip()} | Error: {e}")
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
    return pairs

def write_results(jsonl_path: str, save_dir: str, scores: Dict[Pair, Any], category: str):
    save_file = os.path.join(save_dir, os.path.basename(jsonl_path))
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    try:
        with open(save_file, 'w', encoding='utf-8') as f_out:
            with open(jsonl_path, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    json_line = json.loads(line)
                    instruction = json_line['instruction']
                    input_image = json_line['input_images'][0]
                    output_image1 = json_line['output_images'][0]
                    output_image2 = json_line['output_images'][1]

                    pair1 = (instruction, input_image, output_image1)
                    pair2 = (instruction, input_image, output_image2)
                    
                    score1 = scores.get(pair1, {}).get(category)
                    score2 = scores.get(pair2, {}).get(category)
                    
                    result_data = {
                        'task_type': os.path.basename(os.path.dirname(os.path.dirname(jsonl_path))),
                        'instruction': instruction,
                        'input_images': json_line['input_images'],
                        'output_images': json_line['output_images'],
                        'score': [score1, score2],
                    }
                    f_out.write(json.dumps(result_data, ensure_ascii=False) + '\n')
    except Exception as e:
        logging.error(f"Failed to write results for {jsonl_path}. Error: {e}")

def process_single_item(item, vie_score):
    instruction = item[0]
    input_image_path = item[1]
    output_image_path = item[2]

    input_image = Image.open(input_image_path).convert("RGB")

    output_image = Image.open(output_image_path).convert("RGB")
    output_image = output_image.resize((input_image.size[0], input_image.size[1]))

    score = vie_score.evaluate([input_image, output_image], instruction)
    return item, score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_dir", type=str, default="Edit-Reward-Bench/labeled_pairs")
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument(
        "--backbone", type=str, default="openai", choices=["openai", "qwen25vl", "internvl3_5"]
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="gpt-4.1"
    )
    parser.add_argument(
        "--openai_url", type=str, default="https://api.openai.com/v1/chat/completions"
    )
    parser.add_argument(
        "--key", type=str, default="PUT YOUR API KEY HERE"
    )
    parser.add_argument("--num_pass", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_workers", type=int, default=20)
    parser.add_argument("--score_range", type=int, default=25)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=1536)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--max_num_batched_tokens", type=int, default=1536)
    return parser.parse_args()

def main(args):
    scorer = VIEScore(
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
    )

    cache_dir = os.path.join(args.result_dir, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f"{args.backbone}_{args.model_name_or_path.replace('/', '_')}.jsonl"
    )
    cache_manager = CacheManager(cache_file)

    task_types = sorted(
        [
            d
            for d in os.listdir(args.benchmark_dir)
            if os.path.isdir(os.path.join(args.benchmark_dir, d))
        ]
    )

    for task_type in task_types:
        logging.info(f"--- Processing task type: {task_type} ---")
        task_type_dir = os.path.join(args.benchmark_dir, task_type)

        jsonl_files_to_process = []
        for category in SCORE_CATEGORIES:
            category_dir = os.path.join(task_type_dir, category)
            jsonl_files_to_process.extend(
                glob.glob(os.path.join(category_dir, "*.jsonl"))
            )

        unique_pairs = load_pairs_from_paths(jsonl_files_to_process)

        all_scores = {}
        pairs_to_process = [
            pair
            for pair in unique_pairs
            if cache_manager.get(generate_cache_key(pair)) is None
        ]

        for pair in unique_pairs:
            if pair not in pairs_to_process:
                all_scores[pair] = cache_manager.get(generate_cache_key(pair))

        logging.info(
            f"{len(unique_pairs) - len(pairs_to_process)} pairs found in cache. Processing {len(pairs_to_process)} new pairs."
        )

        if pairs_to_process:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [
                    executor.submit(process_single_item, pair, scorer)
                    for pair in pairs_to_process
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    unit="pair",
                    desc=f"Processing {task_type}",
                ):
                    pair, result = future.result()
                    if result:
                        all_scores[pair] = result
                        cache_manager.append(generate_cache_key(pair), result)

        logging.info(f"Writing results for task '{task_type}'...")
        for category in SCORE_CATEGORIES:
            category_dir = os.path.join(task_type_dir, category)
            jsonl_files = sorted(glob.glob(os.path.join(category_dir, "*.jsonl")))

            save_dir = os.path.join(args.result_dir, args.backbone, task_type, category)

            for jsonl_file in jsonl_files:
                write_results(jsonl_file, save_dir, all_scores, category)

    logging.info("--- All tasks completed! ---")

if __name__ == "__main__":
    args = parse_args()
    main(args)