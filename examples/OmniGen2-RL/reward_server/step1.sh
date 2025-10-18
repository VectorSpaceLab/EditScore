# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# uncomment this if you are using conda
# source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
# conda activate editscore

machine_id=0
model_name=editscore_7B
model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct
lora_path=EditScore/EditScore-7B
score_range=25
tensor_parallel_size=1
max_num_seqs=64
max_model_len=1536
max_num_batched_tokens=98304
num_pass=1

# process named parameters
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name_or_path=*)
            model_name_or_path="${1#*=}"
            shift
            ;;
        --model_name=*)
            model_name="${1#*=}"
            shift
            ;;
        --lora_path=*)
            lora_path="${1#*=}"
            shift
            ;;
        --machine_id=*)
            machine_id="${1#*=}"
            shift
            ;;
        --score_range=*)
            score_range="${1#*=}"
            shift
            ;;
        --tensor_parallel_size=*)
            tensor_parallel_size="${1#*=}"
            shift
            ;;
        --max_num_seqs=*)
            max_num_seqs="${1#*=}"
            shift
            ;;
        --max_model_len=*)
            max_model_len="${1#*=}"
            shift
            ;;
        --max_num_batched_tokens=*)
            max_num_batched_tokens="${1#*=}"
            shift
            ;;
        --num_pass=*)
            num_pass="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            shift
            ;;
    esac
done

export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_LOG_BATCHSIZE_INTERVAL=60

echo ${tensor_parallel_size}

echo $model_name_or_path
echo $lora_path

VLLM_USE_V1=1 VLLM_FLASH_ATTN_VERSION=3 python start_multi_servers.py reward_server --base_port 18888 \
--model_name_or_path ${model_name_or_path} \
--lora_path ${lora_path} \
--tensor_parallel_size ${tensor_parallel_size} \
--num_gpus_per_worker ${tensor_parallel_size} \
--max_num_seqs ${max_num_seqs} \
--max_model_len ${max_model_len} \
--max_num_batched_tokens ${max_num_batched_tokens} \
--score_range ${score_range} \
--num_pass ${num_pass} \
--log_name reward_server_${model_name}_machine${machine_id}