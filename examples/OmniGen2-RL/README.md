# 🚀 Advanced Applications of EditScore

Welcome to the examples directory! Here, we demonstrate how to leverage **EditScore** not just as an evaluation metric, but as a powerful component to actively *improve image editing models*.

This guide covers two primary downstream applications:
1. **Best-of-N selection**: A simple, training-free method to instantly boost the output quality of any image editing model.
2. **Reinforcement Learning (RL) Fine-Tuning**: Using EditScore as a high-fidelity reward signal to train models for significantly better performance.

## 🛠️ Setup for Examples
The examples require libraries for RL, data handling, and potentially experiment tracking.
```bash
# Navigate to this directory if you are in the root
cd examples/OmniGen2-RL

# Install the required packages
pip install -r requirements.txt
```

## Application 1: Best-of-N for Superior Outputs
Best-of-N is an elegant and powerful technique. Instead of generating a single output for a given instruction, you generate multiple (N) candidates and then use a highly accurate evaluator—**EditScore**—to select the best one.

This acts as a powerful "reranker" that filters out suboptimal results, significantly improving the perceived quality of the model without any extra training.

### How to Use
We provide ready-to-use scripts to perform a full Best-of-N workflow on the GEdit-Bench benchmark. The following instructions use **OmniGen2** as the base model, but we provide similar scripts for **FLUX-Kontext** and **Qwen-Image-Edit** in the `evaluation/GEdit-Bench/` directory.

**1. Generate Candidates**
```bash
bash evaluation/GEdit-Bench/omnigen2_16samples.sh # default using 8 GPUs
```

> **⚠️ Important Note on Resource Usage**
>
> This process is computationally expensive and slow due to the large number of generations (16 samples per instruction). For reference, completing this step for OmniGen2 takes approximately **3 hours using 64 H100 GPUs**.

<details>
<summary><strong>👉 Click here for tips on the usage of the script</strong></summary>

- **Distributed Inference**: Our scripts natively support multi-machine and multi-GPU execution. To run inference across 4 machines, for example, execute the following commands on each respective machine:
```bash
# On the first machine (rank 0)
bash evaluation/GEdit-Bench/omnigen2_16samples.sh --world_size 4 --rank 0

# On the second machine (rank 1)
bash evaluation/GEdit-Bench/omnigen2_16samples.sh --world_size 4 --rank 1

# ...and so on for ranks 2 and 3.
```

- **Monitoring Progress**: The scripts utilize nohup for background execution. We recommend monitoring the file (specified in the script file) to track the status and progress of the generation process.
</details>

**2. Score and Select**
Next, use EditScore to evaluate all N candidates and identify the one with the highest score.

```bash
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass1.sh # EditScore-7B, single pass
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass4.sh # EditScore-7B, Avg@4
```

**3. Evaluate the Final Selections**
Finally, evaluate the performance of the images selected by EditScore on GEdit-Bench to quantify the improvement.
```bash
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass1_eval.sh
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass4_eval.sh
```

By comparing these results to the baseline performance of the original model, you will see the benefits of applying EditScore as a reranker.

## Application 2: Reinforcement Fine-Tuning
Beyond evaluation, **EditScore** can be used as a high-quality reward signal to fine-tune your image editing models using Reinforcement Learning (RL), leading to significantly improved performance.

We employ the **FlowGRPO** algorithm, combining its strengths with EditScore's accurate, real-time feedback to create a powerful end-to-end fine-tuning pipeline. This process effectively guides the model toward generating better edits.

### 1. Prepare Training Data
First, set up the dataset for RL fine-tuning.
1. Download the Data
Downlaod the official RL training data from [EditScore-RL-Data](https://huggingface.co/datasets/EditScore/EditScore-RL-Data).
2. Create Meta File
The uploaded dataset uses relative image paths. Run the following script to convert them to absolute paths based on your local environment:
```bash
python scripts/data/process_jsonl.py --input /path/to/EditScore-RL-Data/rl.jsonl --output /path/to/EditScore-RL-Data/rl_abs.jsonl --base-path /path/to/EditScore-RL-Data
```
3. Configure the Data Path
Specify the path to your processed `.jsonl` file in the data configuration located at `data_configs/train/example/edit/all.yml`.
For example:
```yaml
ratio_type: inside_ratio

data:
  - 
    path: '/path/to/EditScore-RL-Data/rl_abs.jsonl' # <-- Ensure this path is correct
    type: 'edit'
    ratio: !!float 1
```

### 2. Prepare the Base Model (OmniGen2)
```bash
python scripts/misc/extract_bin_from_pipe.py
```

### 3. Launch the Reward Server
RL training requires a live reward signal. Before starting the training process, you must launch the **EditScore Reward Server**. This server will provide real-time scores for the generated images during training.

Our reward server is built with two components: a **proxy** and one or more **reward servers**. The proxy receives requests from the training node, distributes them to the individual reward servers for computation, and then collects the results to send back. This architecture allows for easy scaling across multiple machines.

We provide a convenient script to launch the entire server stack across multiple machines, assuming you have `ssh` access to all reward server nodes.

```bash
# Launch EditScore-7B Reward Server
bash reward_server/start_multi_machines.sh --model_name=editscore_7B --config_path=reward_server/server_configs/editscore_7B.yml

# Launch EditScore-7B (Avg@4) Reward Server
bash reward_server/start_multi_machines.sh --model_name=editscore_7B_pass4 --config_path=reward_server/server_configs/editscore_7B_pass4.yml

# Launch EditScore-72B Reward Server
bash reward_server/start_multi_machines.sh --model_name=editscore_72B --config_path=reward_server/server_configs/editscore_72B.yml
```

> **⚠️ Important Notes**
>
> *   Before running the script, you **must** specify the IP addresses of your reward server machines in the corresponding `.yml` configuration file.
> *   If you cannot use `ssh` to control the nodes, please refer to the logic in `reward_server/start_multi_machines.sh` to manually start the proxy and server processes on each machine.
> *   You can monitor the status of the proxy and servers by checking the log files in the `reward_server/logs/` directory.

## 3.5 (Optional) Reward Server Sanity Check
To ensure the reward server is configured correctly and running as expected, we provide a sanity check script.
```bash
python reward_server/scripts/utils/reward_server_sanity_check.py --config_path=reward_server/server_configs/editscore_7B.yml
```
Once these steps are complete, your environment is ready to begin the reinforcement learning fine-tuning process.

### 4. Start Training

**Configure Training Parameters**

Edit the `options/omnigen2_edit_rl.yml` configuration file, focusing on these key parameters:
- `train.global_batch_size`: Global batch size across all GPUs (num_unique_prompts_per_sampling * num_images_per_prompt)
- `train.batch_size`: Batch size per GPU (batch_size_per_forward * gradient_accumulation_steps * num_update_steps_per_sampling)
- `train.rl.num_images_per_prompt`: The number of roolout of one prompt 
- `train.rl.num_unique_prompts_per_sampling`: Number of global unique prompts

**Launch Distributed Training**
```bash
# Single machine training (8*H100 GPUs)
bash scripts/train/omnigen2_edit_rl.sh

# Multi-machine distributed training
```

> **⚠️ Training Configuration Key Points**
>
> **Reward Server IP**: Ensure the `REWARD_SERVER_IP` environment variable in training scripts points to the correct reward server address


### 4. Training Outputs and Monitoring

Logs and saved model checkpoints in `experiments/`