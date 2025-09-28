# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
# cd $(dirname $SHELL_FOLDER)
# cd ../

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.12+pytorch2.7.1+cu126

# export http_proxy=http://10.8.36.1:2080
# export https_proxy=http://10.8.36.1:2080

python evaluation.py \
--benchmark_dir EditScore/EditReward-Bench \
--result_dir results/EditScore-7B \
--backbone qwen25vl \
--model_name_or_path /share/project/jiahao/LLaMA-Factory2/output/merge_v7-2_8models_omnigen2-4samples_gpt4-1_range_0to25 \
--score_range 25 \
--max_workers 1 \
--max_model_len 4096 \
--max_num_seqs 1 \
--max_num_batched_tokens 4096 \
--tensor_parallel_size 1 \
--num_pass 1

python calculate_statistics.py \
--result_dir results/EditScore-7B/qwen25vl