import dotenv

dotenv.load_dotenv(override=True)

import argparse

from omegaconf import OmegaConf

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

from accelerate import init_empty_weights

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


def main(args):
    model_path = args.model_path
    save_path = args.save_path

    dcp_to_torch_save(model_path, save_path)

    state_dict = torch.load(save_path, weights_only=True)['model']

    torch.save(state_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
