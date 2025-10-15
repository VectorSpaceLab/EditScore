# 🚀 Advanced Applications of EditScore

Welcome to the examples directory! Here, we demonstrate how to leverage **EditScore** not just as an evaluation metric, but as a powerful component to actively *improve image editing models*.

This guide covers two primary downstream applications:
1. **Best-of-N selection**: A simple, training-free method to instantly boost the output quality of any image editing model.
2. **Reinforcement Learning (RL) Fine-Tuning**: Using EditScore as a high-fidelity reward signal to train models for significantly better performance.

## 🛠️ Setup for Examples
The examples require libraries for RL (trl, peft), data handling, and potentially experiment tracking.
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

**1.Generate Candidates**
```bash
bash evaluation/GEdit-Bench/omnigen2_16samples.sh
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

- **GPU Configuration**: You can adjust the number of GPUs used per machine by modifying the num_gpus_per_machine variable inside the script.
- **Monitoring Progress**: The scripts utilize nohup for background execution. We recommend monitoring the file (specified in the script file) to track the status and progress of the generation process.

**2.Score and Select**
Next, use EditScore to evaluate all N candidates and identify the one with the highest score.

```bash
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass1.sh # EditScore-7B, single pass
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass4.sh # EditScore-7B, Avg@4
```

**3.Evaluate**
Next, evaluate on GEdit-Bench see the benefits of applying EditScore.
```bash
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass1_eval.sh
bash evaluation/GEdit-Bench/omnigen2_16samples_select_best_editscore_pass4_eval.sh
```