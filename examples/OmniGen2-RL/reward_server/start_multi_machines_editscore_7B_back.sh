# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

job_id=job-850d72d4-438b-4dce-8399-820d92fa2f0f

# ip address of the machines
machines=(
$job_id-master-0
$job_id-worker-0
)

model_name=editscore_7B
model_name_or_path=Qwen/Qwen2.5-VL-7B-Instruct
lora_path=EditScore/EditScore-7B

root_dir=$SHELL_FOLDER

SCRIPT="bash ${root_dir}/step1.sh --num_pass=1 --score_range=25 --tensor_parallel_size=1 --max_num_seqs=64 --max_model_len=1536 --max_num_batched_tokens=98304 --model_name=${model_name} --model_name_or_path=${model_name_or_path} --lora_path=${lora_path}"

for i in "${!machines[@]}"; do
    echo "$i ${machines[$i]}"
    ssh -o StrictHostKeyChecking=no "${machines[$i]}" "tmux new-session -d -s reward_server '$SCRIPT --machine_id=$i'" &
done

SCRIPT="bash ${root_dir}/step2.sh --max_workers_per_machine=8 --model_name=${model_name}"

for i in "${!machines[@]}"; do
    echo "$i ${machines[$i]}"
    ssh -o StrictHostKeyChecking=no "${machines[i]}" "tmux new-session -d -s reward_proxy '$SCRIPT --worker_host=${job_id} --worker_num_machines=${#machines[@]} --machine_id=$i'" &
done

echo "step2.sh started successfully"

wait
echo "All tasks started successfully"