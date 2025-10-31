# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

python evaluation.py \
--benchmark_dir EditScore/EditReward-Bench \
--result_dir results/EditScore-Qwen3-VL-8B-avg4 \
--backbone qwen3vl \
--model_name_or_path Qwen/Qwen3-VL-8B-Instruct \
--lora_path EditScore/EditScore-Qwen3-VL-8B-Instruct \
--score_range 25 \
--max_workers 1 \
--max_model_len 4096 \
--max_num_seqs 1 \
--max_num_batched_tokens 4096 \
--tensor_parallel_size 1 \
--num_pass 4

python calculate_statistics.py \
--result_dir results/EditScore-Qwen3-VL-8B-avg4/qwen3vl