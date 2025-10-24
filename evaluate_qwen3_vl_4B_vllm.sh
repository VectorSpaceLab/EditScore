# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate editscore

python evaluation.py \
--benchmark_dir EditScore/EditReward-Bench \
--result_dir results/EditScore-Qwen3-VL-4B \
--backbone qwen3vl_vllm \
--model_name_or_path /share/project/shared_models/Qwen3-VL-4B-Instruct \
--lora_path /share/project/jiahao/LLaMA-Factory3/output/editscore_qwen3_4B_ins \
--score_range 25 \
--max_workers 1 \
--max_model_len 4096 \
--max_num_seqs 1 \
--max_num_batched_tokens 4096 \
--tensor_parallel_size 1 \
--num_pass 1

python calculate_statistics.py \
--result_dir results/EditScore-Qwen3-VL-4B/qwen3vl_vllm