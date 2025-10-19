# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
# cd $(dirname $SHELL_FOLDER)

job_id=job-09ff3d74-0547-4085-a7fb-92ecd6e9b1e0
num_machines=2

# 处理命名参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --job_id=*)
            job_id="${1#*=}"
            shift
            ;;
        --num_machines=*)
            num_machines="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            shift
            ;;
    esac
done

machines=(
$job_id-master-0
)

echo $num_machines

for i in $(seq 0 $((num_machines - 1))); do
    machines+=("$job_id-worker-$i")
done

echo "machines: ${machines[@]}"

SCRIPT="pkill python"

for i in "${!machines[@]}"; do
    echo "$i ${machines[$i]}"
    ssh -o StrictHostKeyChecking=no "${machines[$i]}" "$SCRIPT" &
done

wait
echo "All tasks in remote machines killed"