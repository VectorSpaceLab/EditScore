import dotenv

dotenv.load_dotenv(override=True)

import os
import sys
import json
from PIL import Image
import random

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from omnigen2.grpo.reward_client_edit import evaluate_images
from datasets import load_from_disk


def test_reward_server():
    N = 48
    K = 12
    images = [
        Image.open(
            "/share/project/luoxin/projects2/OmniGen2-RL/evaluation/GEdit-Bench/results/OmniGen2_pass16/results_ts5.0_ig1.5/fullset/background_change/en/0f385bcff859231789a9c978cafecc2a_SRCIMG.png"
        ).resize((512, 512))
    ] * (N * K)
    input_images = [
        [
            Image.open(
                "/share/project/luoxin/projects2/OmniGen2-RL/evaluation/GEdit-Bench/results/OmniGen2_pass16/results_ts5.0_ig1.5/fullset/background_change/en/0f385bcff859231789a9c978cafecc2a.png"
            ).resize((512, 512))
        ]
    ] * (N * K)
    meta_datas = [json.dumps({"instruction": f"Change the background to a starry sky.{i}"}) for i in range(K) for j in range(N)]

    # images = []
    # input_images = []
    # meta_datas = []

    # test_dataset = load_from_disk("/share/project/chenyuan/data/GEdit-Bench")
    # # 根据规则过滤test_dataset
    # filtered_test_dataset = [
    #     item for item in test_dataset
    #     if item['instruction_language'] != 'cn'
    #     # and item['task_type'] not in ['material_alter', 'ps_human', 'motion_change', 'text_change', 'tone_transfer']
    # ]
    # test_dataset = filtered_test_dataset

    # data_index = list(range(0, 2))

    # for idx in data_index:
    #     for i in range(4):
    #         data_item = test_dataset[idx]

    #         task_type = data_item['task_type']
    #         instruction_language = data_item['instruction_language']

    #         key = data_item['key']
    #         instruction = data_item['instruction']

    #         sub_dir = os.path.join("/share/project/luoxin/projects2/OmniGen2-RL/evaluation/GEdit-Bench/results/OmniGen2_pass16/results_ts5.0_ig1.5", "fullset", task_type, instruction_language)

    #         input_images.append([Image.open(os.path.join(sub_dir, f"{key}_SRCIMG.png")).resize((512, 512))])
    #         images.append(Image.open(os.path.join(sub_dir, f"{key}.png")).resize((512, 512)))
    #         meta_datas.append(json.dumps({"instruction": instruction}))

    # permuted_idx = list(range(len(input_images)))
    # random.shuffle(permuted_idx)
    # input_images = [input_images[i] for i in permuted_idx]
    # images = [images[i] for i in permuted_idx]
    # meta_datas = [meta_datas[i] for i in permuted_idx]
            
    scores, rewards, reasoning, meta_data = evaluate_images(
        input_images=input_images, 
        output_image=images,
        meta_datas=meta_datas,
        proxy_host="job-850d72d4-438b-4dce-8399-820d92fa2f0f-master-0",
        server_type='vlm'
    )
    print(scores, rewards, reasoning, meta_data)


if __name__ == "__main__":
    test_reward_server()