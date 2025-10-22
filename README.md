<p align="center">
  <img src="https://raw.githubusercontent.com/VectorSpaceLab/EditScore/refs/heads/main/assets/logo.png" width="65%">
</p>

<p align="center">
  <a href="https://vectorspacelab.github.io/EditScore"><img src="https://img.shields.io/badge/Project%20Page-EditScore-yellow" alt="project page"></a>
  <a href="https://arxiv.org/abs/2509.23909"><img src="https://img.shields.io/badge/arXiv%20paper-2509.23909-b31b1b.svg" alt="arxiv"></a>
  <a href="https://huggingface.co/collections/EditScore/editscore-68d8e27ee676981221db3cfe"><img src="https://img.shields.io/badge/EditScore-🤗-yellow" alt="model"></a>
  <a href="https://huggingface.co/datasets/EditScore/EditReward-Bench"><img src="https://img.shields.io/badge/EditReward--Bench-🤗-yellow" alt="dataset"></a>
  <a href="https://huggingface.co/datasets/EditScore/EditScore-Reward-Data"><img src="https://img.shields.io/badge/EditScore--Reward--Data-🤗-yellow" alt="dataset"></a>
  <a href="https://huggingface.co/datasets/EditScore/EditScore-RL-Data"><img src="https://img.shields.io/badge/EditScore--RL--Data-🤗-yellow" alt="dataset"></a>
</p>

<h4 align="center">
    <p>
        <a href=#-news>News</a> |
        <a href=#-quick-start>Quick Start</a> |
        <a href=#-benchmark-your-image-editing-reward-model usage>Benchmark Usage</a> |
        <a href=#%EF%B8%8F-citing-us>Citation</a>
    <p>
</h4>

**EditScore** is a series of state-of-the-art open-source reward models (7B–72B) designed to evaluate and enhance instruction-guided image editing.
## ✨ Highlights
- **State-of-the-Art Performance**: Effectively matches the performance of leading proprietary VLMs. With a self-ensembling strategy, **our largest model surpasses even GPT-5** on our comprehensive benchmark, **EditReward-Bench**.
- **A Reliable Evaluation Standard**: We introduce **EditReward-Bench**, the first public benchmark specifically designed for evaluating reward models in image editing, featuring 13 subtasks, 11 state-of-the-art editing models (*including proprietary models*) and expert human annotations.
- **Simple and Easy-to-Use**: Get an accurate quality score for your image edits with just a few lines of code.
- **Versatile Applications**: Ready to use as a best-in-class reranker to improve editing outputs, or as a high-fidelity reward signal for **stable and effective Reinforcement Learning (RL) fine-tuning**.

## 🔥 News
- **2025-10-22**: **Introducing Our Reinforcement Learning Training Framework!**
  We're excited to release our complete RL pipeline, the result of a massive effort to simplify fine-tuning for image editing models. Key features include:
    - **Ready-to-Use RL Dataset**: Includes the complete dataset used in the EditScore project, along with clear usage guidelines and preparation scripts.
    - **An Easy-to-Use Reward Model**: Seamlessly integrate **EditScore** as a reward signal.
    - **A Scalable Reward Server**: Built with native multi-node support for high-throughput training.
    - **Flexible Training Code**: Supports distributed training, variable image resolutions and mixed tasks (t2i, edit, in-context generation) out-of-the-box.
  Dive into our comprehensive guide on [RL Fine-Tuning](examples/OmniGen2-RL#application-2-reinforcement-fine-tuning) to get started.

- 2025-10-16: Training datasets [EditScore-Reward-Data](https://huggingface.co/datasets/EditScore/EditScore-Reward-Data) and [EditScore-RL-Data](https://huggingface.co/datasets/EditScore/EditScore-RL-Data) are available.
- 2025-10-15: **EditScore** is now available on PyPI — install it easily with `pip install editscore`.
- 2025-10-15: Best-of-N inference scripts for OmniGen2, Flux-dev-Kontext, and Qwen-Image-Edit are now available! See [this](#apply-editscore-to-image-editing) for details.
- 2025-09-30: We release **OmniGen2-EditScore7B**, unlocking online RL For Image Editing via high-fidelity EditScore. LoRA weights are available at [Hugging Face](https://huggingface.co/OmniGen2/OmniGen2-EditScore7B) and [ModelScope](https://www.modelscope.cn/models/OmniGen2/OmniGen2-EditScore7B).
- 2025-09-30: We are excited to release **EditScore** and **EditReward-Bench**! Model weights and the benchmark dataset are now publicly available. You can access them on Hugging Face: [Models Collection](https://huggingface.co/collections/EditScore/editscore-68d8e27ee676981221db3cfe) and [Benchmark Dataset](https://huggingface.co/datasets/EditScore/EditReward-Bench), and on ModelScope: [Models Collection](https://www.modelscope.cn/collections/EditScore-8b0d53aa945d4e) and [Benchmark Dataset](https://www.modelscope.cn/datasets/EditScore/EditReward-Bench).

## 📖 Introduction
While Reinforcement Learning (RL) holds immense potential for this domain, its progress has been severely hindered by the absence of a high-fidelity, efficient reward signal.

To overcome this barrier, we provide a systematic, two-part solution:

- **A Rigorous Evaluation Standard**: We first introduce **EditReward-Bench**, a new public benchmark for the direct and reliable evaluation of reward models. It features 13 diverse subtasks and expert human annotations, establishing a gold standard for measuring reward signal quality.

- **A Powerful & Versatile Tool**: Guided by our benchmark, we developed the **EditScore** model series. Through meticulous data curation and an effective self-ensembling strategy, EditScore sets a new state of the art for open-source reward models, even surpassing the accuracy of leading proprietary VLMs.

<p align="center">
  <img src="https://raw.githubusercontent.com/VectorSpaceLab/EditScore/refs/heads/main/assets/table_reward_model_results.png" width="95%">
  <br>
  <em>Benchmark results on EditReward-Bench.</em>
</p>

We demonstrate the practical utility of EditScore through two key applications:

- **As a State-of-the-Art Reranker**: Use EditScore to perform Best-of-*N* selection and instantly improve the output quality of diverse editing models.
- **As a High-Fidelity Reward for RL**: Use EditScore as a robust reward signal to fine-tune models via RL, enabling stable training and unlocking significant performance gains where general-purpose VLMs fail.

This repository releases both the **EditScore** models and the **EditReward-Bench** dataset to facilitate future research in reward modeling, policy optimization, and AI-driven model improvement.

<p align="center">
  <img src="https://raw.githubusercontent.com/VectorSpaceLab/EditScore/refs/heads/main/assets/figure_edit_results.png" width="95%">
  <br>
  <em>EditScore as a superior reward signal for image editing.</em>
</p>


## 📌 TODO
We are actively working on improving EditScore and expanding its capabilities. Here's what's next:


- [x] Release training data for reward model and online RL.
- [x] Release RL training code applying EditScore to OmniGen2.
- [x] Provide Best-of-N inference scripts for OmniGen2, Flux-dev-Kontext, and Qwen-Image-Edit.

## 🚀 Quick Start

### 🛠️ Environment Setup
We offer two ways to install EditScore. Choose the one that best fits your needs.
**Method 1: Install from PyPI (Recommended for Users)**: If you want to use EditScore as a library in your own project.
**Method 2: Install from Source (For Developers)**: If you plan to contribute to the code, modify it, or run the examples in this repository

#### Prerequisites: Installing PyTorch
Both installation methods require PyTorch to be installed first, as its version is dependent on your system's CUDA setup.
```bash
# (Optional) Create a clean Python environment
conda create -n editscore python=3.12
conda activate editscore

# Choose the command that matches your CUDA version.
# This example is for CUDA 12.6.
pip install torch==2.7.1 torchvision --extra-index-url https://download.pytorch.org/whl/cu126
````

<details>
<summary>🌏 For users in Mainland China</summary>
```bash
# Install PyTorch from a domestic mirror
pip install torch==2.7.1 torchvision --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu126
```
</details>

#### Method 1: Install from PyPI (Recommended for Users)
```bash
pip install -U editscore
```

#### Method 2: Install from Source (For Developers)
This method gives you a local, editable version of the project.
1. Clone the repository
```bash
git clone https://github.com/VectorSpaceLab/EditScore.git
cd EditScore
```

2. Install EditScore in editable mode
```bash
pip install -e .
```

#### ✅ (Recommended) Install Optional High-Performance Dependencies
For the best performance, especially during inference, we highly recommend installing vllm.
```bash
pip install vllm
```

---

### 🧪 Usage Example
Using EditScore is straightforward. The model will be automatically downloaded from the Hugging Face Hub on its first run.
```python
from PIL import Image
from editscore import EditScore

# Load the EditScore model. It will be downloaded automatically.
# Replace with the specific model version you want to use.
model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
lora_path = "EditScore/EditScore-7B"

scorer = EditScore(
    backbone="qwen25vl", # set to "qwen25vl_vllm" for faster inference
    model_name_or_path=model_path,
    enable_lora=True,
    lora_path=lora_path,
    score_range=25,
    num_pass=1, # Increase for better performance via self-ensembling
)

input_image = Image.open("example_images/input.png")
output_image = Image.open("example_images/output.png")
instruction = "Adjust the background to a glass wall."

result = scorer.evaluate([input_image, output_image], instruction)
print(f"Edit Score: {result['final_score']}")
# Expected output: A dictionary containing the final score and other details.
```

---

## 📊 Benchmark Your Image-Editing Reward Model
#### Install benchmark dependencies
To use example code for benchmark, run following
```bash
pip install -r requirements.txt
```

We provide an evaluation script to benchmark reward models on **EditReward-Bench**. To evaluate your own custom reward model, simply create a scorer class with a similar interface and update the script.
```bash
# This script will evaluate the default EditScore model on the benchmark
bash evaluate.sh

# Or speed up inference with VLLM
bash evaluate_vllm.sh
```

## Apply EditScore to Image Editing
We offer two example use cases for your exploration:
- **Best-of-N selection**: Use EditScore to automatically pick the most preferred image among multiple candidates.
- **Reinforcement fine-tuning**: Use EditScore as a reward model to guide RL-based optimization.

For detailed instructions and examples, please refer to the [documentation](examples/OmniGen2-RL/README.md).

## ❤️ Citing Us
If you find this repository or our work useful, please consider giving a star ⭐ and citation 🦖, which would be greatly appreciated:

```bibtex
@article{luo2025editscore,
  title={EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling},
  author={Xin Luo and Jiahao Wang and Chenyuan Wu and Shitao Xiao and Xiyan Jiang and Defu Lian and Jiajun Zhang and Dong Liu and Zheng Liu},
  journal={arXiv preprint arXiv:2509.23909},
  year={2025}
}
```
