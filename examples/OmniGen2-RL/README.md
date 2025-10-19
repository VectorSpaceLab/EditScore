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
Use EditScore to provide a high-quality reward signal to train models for significantly better image editing performance. We employ the FlowGRPO algorithm combined with EditScore's accurate evaluation capabilities to achieve end-to-end reinforcement learning fine-tuning.

### 1. Data and Model Download
Download RL training data from [EditScore-RL-Data](https://huggingface.co/datasets/EditScore/EditScore-RL-Data), then put the `rl.jsonl` into `data/` and change its path in `data_configs/train/train.yml`

Download the base model OmniGen2 form [OmniGen2](https://huggingface.co/OmniGen2/OmniGen2),then change the model file format to pytorch_model.bin and modify `model.pretrained_model_path` in `options/omnigen2_edit_rl.yml`

### 2. Start Reward Server

Before beginning training, you need to start the EditScore reward server to provide real-time reward signal evaluation for RL training.

### 3. Start Training

**Configure Training Parameters**

Edit the `options/omnigen2_edit_rl.yml` configuration file, focusing on these key parameters:
- `train.global_batch_size`: Global batch size (num_machines * num_unique_prompts_per_sampling * num_images_per_prompt)
- `train.rl.num_images_per_prompt`: Rollout number of one prompt 
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