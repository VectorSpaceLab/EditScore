import dotenv

dotenv.load_dotenv(override=True)

import os
import sys
import json
from PIL import Image
import argparse
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from omnigen2.grpo.reward_client_edit import evaluate_images


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    return parser.parse_args()

def main(args):
    config = yaml.load(open(args.config_path, "r"), Loader=yaml.FullLoader)
    proxy_host = config['server']['hosts'][0]
    proxy_port = config['server']['proxy_port']

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
    meta_datas = [
        json.dumps({"instruction": f"Change the background to a starry sky.{i}"})
        for i in range(K)
        for j in range(N)
    ]

    scores, rewards, reasoning, meta_data = evaluate_images(
        input_images=input_images,
        output_image=images,
        meta_datas=meta_datas,
        proxy_host=proxy_host,
        proxy_port=proxy_port,
        server_type="vlm",
    )
    print(scores, rewards, reasoning, meta_data)


if __name__ == "__main__":
    args = parse_args()
    main(args)