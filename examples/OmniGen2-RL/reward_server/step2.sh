# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER

# uncomment this if you are using conda
# source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
# conda activate editscore

worker_host=job-96f2e55a-f22a-42d5-9a45-c81edeb54e88
worker_num_machines=2
machine_id=0
model_name=editscore_7B
max_workers_per_machine=8


while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name=*)
            model_name="${1#*=}"
            shift
            ;;
        --worker_host=*)
            worker_host="${1#*=}"
            shift
            ;;
        --worker_num_machines=*)
            worker_num_machines="${1#*=}"
            shift
            ;;
        --machine_id=*)
            machine_id="${1#*=}"
            shift
            ;;
        --max_workers_per_machine=*)
            max_workers_per_machine="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            shift
            ;;
    esac
done

python reward_proxy.py --worker_host $worker_host --worker_base_port 18888 --worker_num_machines $worker_num_machines --max_workers_per_machine $max_workers_per_machine \
>logs/reward_proxy_${model_name}_workers${max_workers_per_machine}_machine${machine_id}.log 2>&1