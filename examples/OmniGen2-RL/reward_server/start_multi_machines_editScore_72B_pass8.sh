# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

job_id=job-f2c6423e-7908-486d-b59f-dc82faaf2065

# ip address of the machines
machines=(
$job_id-master-0
$job_id-worker-0
$job_id-worker-1
$job_id-worker-2
$job_id-worker-3
$job_id-worker-4
)

model_name=EditScore_72B_pass8
model_name_or_path=Qwen/Qwen2.5-VL-72B-Instruct
lora_path=EditScore/EditScore-72B

root_dir=$SHELL_FOLDER

SCRIPT="bash ${root_dir}/step1.sh --num_pass=8 --score_range=25 --tensor_parallel_size=4 --max_num_seqs=64 --max_model_len=1536 --max_num_batched_tokens=98304"

for i in "${!machines[@]}"; do
    echo "$i ${machines[$i]}"
    ssh -o StrictHostKeyChecking=no "${machines[$i]}" "tmux new-session -d -s reward_server '$SCRIPT --machine_id=$i --model_name=${model_name} --model_name_or_path=${model_name_or_path} --lora_path=${lora_path}'" &
done

SCRIPT="bash ${root_dir}/step2.sh --max_workers_per_machine=2"

for i in "${!machines[@]}"; do
    echo "$i ${machines[$i]}"
    ssh -o StrictHostKeyChecking=no "${machines[i]}" "tmux new-session -d -s reward_proxy '$SCRIPT --worker_host=${job_id} --worker_num_machines=${#machines[@]} --machine_id=$i --model_name=${model_name}'" &
done

echo "step2.sh started successfully"

wait
echo "All tasks started successfully"