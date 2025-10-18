#!/usr/bin/env python3

from typing import List
import argparse
import pickle
import json
import os
import warnings
import threading
from queue import Queue
from typing import Dict, Tuple
import uuid
import time

from flask import Flask, request, jsonify
from PIL import Image

from editscore import EditScore

warnings.filterwarnings("ignore")

app = Flask(__name__)

# --- Global queue and result storage ---
request_queue = Queue()
results = {} # Use a dict to store results, associated by unique ID

def apply_chat_template(prompt, num_images: int = 2):
    """
    This is used since the bug of transformers which do not support vision id https://github.com/QwenLM/Qwen2.5-VL/issues/716#issuecomment-2723316100
    """
    template = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    template += "".join([f"<img{i}>: <|vision_start|><|image_pad|><|vision_end|>" for i in range(1, num_images + 1)])
    template += f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return template

class VLMScorer:
    """Encapsulates vLLM model and scoring logic."""
    def __init__(self, args: argparse.Namespace):
        print("🔧 Initializing VLMScorer...")
        self.scorer = EditScore(
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
            seed=args.seed,
        )
        print("✅ VLMScorer initialization complete.")

    def score(self, input_images: List[List[Image.Image]], output_image: List[Image.Image], metadata: Dict[str, any]) -> float:
        """Score a batch of samples."""
        
        image_prompts = []
        for input_image, _output_image in zip(input_images, output_image):
            image_prompts.append(input_image + [_output_image])
            
        results = self.scorer.batch_evaluate(image_prompts, [_metadata['instruction'] for _metadata in metadata])

        outputs = []
        for result in results:
            reward = result['O_score'] / 10
            reasoning = f"SC_score: {result['SC_score']}\n"
            reasoning += f"SC_score_reasoning: {result['SC_score_reasoning']}\n"
            reasoning += f"PQ_score: {result['PQ_score']}\n"
            reasoning += f"PQ_score_reasoning: {result['PQ_score_reasoning']}\n"
            reasoning += f"SC_raw_output: {result['SC_raw_output']}\n"
            reasoning += f"PQ_raw_output: {result['PQ_raw_output']}\n"
            outputs.append((reward, reasoning))
        return outputs

def vlm_worker(scorer: VLMScorer):
    """Background worker thread, continuously fetches and processes tasks from the queue."""
    print("🚀 VLM background worker thread started, waiting for tasks...")
    while True:
        try:
            task_id, input_images, output_image, meta_data = request_queue.get()
            
            # print(f"🔩 Start processing task {task_id[:8]}...")
            outputs = scorer.score(input_images, output_image, meta_data)
            result_payload = []
            for (reward, reasoning), _meta_data in zip(outputs, meta_data):
                result_payload.append(
                    {
                        "score": 1.0 if reward >= 0.5 else 0.0,
                        "reward": reward,
                        "reasoning": reasoning,
                        "strict_reward": reward,
                        "meta_data": _meta_data,
                        "group_reward": {_meta_data.get("tag", "vlm"): reward},
                        "group_strict_reward": {_meta_data.get("tag", "vlm"): reward},
                    }
                )
            results[task_id] = pickle.dumps(result_payload)

        except Exception as e:
            print(f"❌ Worker thread error while processing task {task_id[:8]}: {e}")
            import traceback
            traceback.print_exc()
            error_result = {"error": f"Internal server error: {e}"}
            results[task_id] = pickle.dumps(error_result)
        finally:
            request_queue.task_done()

# --- Web layer (Flask App) ---

def parse_and_validate_request(raw_data: bytes) -> Tuple[List[Image.Image], Image.Image, Dict, str]:
    """Parse request data, validate and convert to required format."""
    try:
        data = pickle.loads(raw_data)
        input_images_datas = data['input_images']
        output_image_datas = data['output_image']
        meta_data = data['meta_data']
    except Exception as e:
        print(f"Failed to parse request data: {e}")
        return None, None, None, f"Failed to parse request data: {e}"
    
    batch_output_image = []
    for output_image_data in output_image_datas:
        batch_output_image.append(output_image_data.convert('RGB'))

    batch_input_images = []
    for input_image_data in input_images_datas:
        batch_input_images.append([])
        for _input_image_data in input_image_data:
            batch_input_images[-1].append(_input_image_data.convert('RGB'))
    
    batch_meta_data = []
    for _meta_data in meta_data:
        if isinstance(_meta_data, str):
            try:
                _meta_data = json.loads(_meta_data)
            except json.JSONDecodeError:
                _meta_data = {'prompt': _meta_data}

        if not isinstance(_meta_data, dict):
            return None, None, None, f"Meta data must be a dict or JSON string"
        batch_meta_data.append(_meta_data)
    return batch_input_images, batch_output_image, batch_meta_data, None

@app.route('/', methods=['POST'])
def evaluate_batch_samples():
    """Receive request, put it into the queue, and wait for the result to return."""
    
    input_images, output_image, meta_data, error_msg = parse_and_validate_request(request.data)
    if error_msg:
        print(f"❌ Request validation failed: {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    task_id = str(uuid.uuid4())
    request_queue.put((task_id, input_images, output_image, meta_data))
    print(f"📥 Task {task_id[:8]} enqueued, {len(input_images)=}, {len(output_image)=}, {len(meta_data)=}, current queue size: {request_queue.qsize()}", flush=True)

    timeout_seconds = 600
    start_time = time.time()

    while True:
        if task_id in results:
            result_data = results.pop(task_id)
            print(f"📤 Task {task_id[:8]} result returned. Time elapsed: {time.time() - start_time:.2f}s")
            return result_data, 200, {'Content-Type': 'application/octet-stream'}
        
        if time.time() - start_time > timeout_seconds:
            print(f"⌛️ Task {task_id[:8]} timed out waiting.")
            return jsonify({"error": "Request timed out"}), 504
            
        time.sleep(0.05)


def arg_parser():
    parser = argparse.ArgumentParser(description='VLM Reward Server - High concurrency optimized (Flask native server)')
    parser.add_argument('--port', type=int, default=18096, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host (0.0.0.0 means listen on all interfaces)')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    parser.add_argument(
        "--backbone",
        type=str,
        default="qwen25vl_vllm",
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
    args = parser.parse_args()
    return args

def main(args):
    """Main function, loads model, starts background worker thread and web server."""
    # setup_gpu_env(args.gpu_id)

    # 1. Load model
    print("⚡ Preloading VLM model...")
    scorer = VLMScorer(args)
    
    # 2. Start background worker thread
    worker_thread = threading.Thread(target=vlm_worker, args=(scorer,), daemon=True)
    worker_thread.start()

    # 3. Start Flask web server
    print(f"🔥 Starting VLM reward server at http://{args.host}:{args.port}")
    print("🚀 Mode: High concurrency single-sample requests (queue-based processing)")
    print(f"{args.score_range=} {args.tensor_parallel_size=}")
    
    # Use Flask's built-in development server with threading enabled
    try:
        # threaded=True allows the server to handle multiple HTTP requests simultaneously
        # use_reloader=False is necessary when using background threads to prevent the reloader from creating duplicate threads and model instances
        app.run(host=args.host, port=args.port, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 VLM server stopped.")
    except Exception as e:
        print(f"❌ VLM server failed to start: {e}")

if __name__ == '__main__':
    args = arg_parser()
    main(args)