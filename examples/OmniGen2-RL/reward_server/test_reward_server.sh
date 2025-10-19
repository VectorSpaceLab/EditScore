# !/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $(dirname $SHELL_FOLDER)

source "$(dirname $(which conda))/../etc/profile.d/conda.sh"
conda activate py3.12+pytorch2.7.1+cu126

reward_server_ip="job-850d72d4-438b-4dce-8399-820d92fa2f0f-master-0"
# reward_server_ip="6ec80bfc-5d9a-4576-8506-19f781034447-master-0"
export REWARD_SERVER_IP=$reward_server_ip

no_proxy="localhost,127.0.0.1,::1,$reward_server_ip,*.svc.cluster.local" \
python scripts/test_reward_server.py