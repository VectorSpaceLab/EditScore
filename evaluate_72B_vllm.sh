# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.12+pytorch2.7.1+cu126

python evaluation.py \
--benchmark_dir EditScore/EditReward-Bench \
--result_dir results/EditScore-72B \
--backbone qwen25vl_vllm \
--model_name_or_path Qwen/Qwen2.5-VL-72B-Instruct \
--enable_lora \
--lora_path /share/project/jiahao/LLaMA-Factory2/output/lora_72B_extract \
--score_range 25 \
--max_workers 1 \
--max_model_len 4096 \
--max_num_seqs 1 \
--max_num_batched_tokens 4096 \
--tensor_parallel_size 4 \
--num_pass 1

python calculate_statistics.py \
--result_dir results/EditScore-72B/qwen25vl_vllm