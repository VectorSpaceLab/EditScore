import os
import glob
import json
import numpy as np

import argparse

PROMPT_FOLLOWING = "prompt_following"
CONSISTENCY = "consistency"
OVERALL = "overall"
SCORE_CATEGORIES = [PROMPT_FOLLOWING, CONSISTENCY, OVERALL]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="qwen25vl", choices=["qwen25vl", "openai", "internvl3_5"])
    return parser.parse_args()

def main(args):
    task_types = sorted(os.listdir(args.result_dir))

    print(task_types)

    prompt_following_results = dict()
    consistency_results = dict()
    overall_results = dict()

    all_prompt_following_scores = []
    all_consistency_scores = []
    all_overall_scores = []

    for task_type in task_types:
        task_type_dir = os.path.join(args.result_dir, task_type)
        prompt_following_json_file = os.path.join(task_type_dir, f"{PROMPT_FOLLOWING}.jsonl")
        consistency_json_file = os.path.join(task_type_dir, f"{CONSISTENCY}.jsonl")
        overall_json_file = os.path.join(task_type_dir, f"{OVERALL}.jsonl")

        total_num = 0
        correct_num = 0
        with open(prompt_following_json_file, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                if json_line['score'][0] > json_line['score'][1]:
                    correct_num += 1
                total_num += 1
                all_prompt_following_scores.append(json_line['score'][0])
                all_prompt_following_scores.append(json_line['score'][1])
        prompt_following_results[task_type] = correct_num / total_num
        
        total_num = 0
        correct_num = 0
        with open(consistency_json_file, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                if json_line['score'][0] > json_line['score'][1]:
                    correct_num += 1
                total_num += 1
                all_consistency_scores.append(json_line['score'][0])
                all_consistency_scores.append(json_line['score'][1])
        consistency_results[task_type] = correct_num / total_num
        
        total_num = 0
        correct_num = 0
        with open(overall_json_file, 'r') as f:
            for line in f:
                json_line = json.loads(line)
                if json_line['score'][0] > json_line['score'][1]:
                    correct_num += 1
                total_num += 1
                all_overall_scores.append(json_line['score'][0])
                all_overall_scores.append(json_line['score'][1])
        overall_results[task_type] = correct_num / total_num

    prompt_following_results['average'] = sum(prompt_following_results.values()) / len(prompt_following_results)
    consistency_results['average'] = sum(consistency_results.values()) / len(consistency_results)
    overall_results['average'] = sum(overall_results.values()) / len(overall_results)

    print(overall_results.keys())

    task_types = [
        'background_change', 'color_alter', 'style_change', 'subject-add', 'subject-remove', 'subject-replace', 'material_alter',
        'motion_change', 'ps_human', 'text_change', 'tone_transfer', 'extract', 'compose', 'average'
    ]

    print(" & ".join(task_types))
    print("Prompt Following: " + " & ".join([f"{prompt_following_results[task_type]:.3f}" for task_type in task_types]))
    print("Consistency: " + " & ".join([f"{consistency_results[task_type]:.3f}" for task_type in task_types]))
    print("Overall: " + " & ".join([f"{overall_results[task_type]:.3f}" for task_type in task_types]))
    
    groups = {
        'object': ['subject-add', 'subject-remove', 'subject-replace'],
        'appearance': ['color_alter', 'material_alter', 'style_change', 'tone_transfer'],
        'scene': ['background_change', 'extract'],
        'advanced': ['ps_human', 'text_change', 'motion_change', 'compose'],
    }

    print("--------------------------------")
    print("--------------------------------")

    for group_name, group_task_types in groups.items():
        print(group_name + ":")
        print("Prompt Following & Consistency & Overall")
        prompt_following_mean = np.mean([prompt_following_results[task_type] for task_type in group_task_types])
        consistency_mean = np.mean([consistency_results[task_type] for task_type in group_task_types])
        overall_mean = np.mean([overall_results[task_type] for task_type in group_task_types])
        print(f"{prompt_following_mean:.3f} & {consistency_mean:.3f} & {overall_mean:.3f}")
    
    print("Average:")
    print("Prompt Following & Consistency & Overall")
    print(f"{prompt_following_results['average']:.3f} & {consistency_results['average']:.3f} & {overall_results['average']:.3f}")

    print("Prompt Following Scores:")
    print("Min & Max & Mean & Std")
    print(f"{np.min(all_prompt_following_scores):.3f} & {np.max(all_prompt_following_scores):.3f} & {np.mean(all_prompt_following_scores):.3f} & {np.std(all_prompt_following_scores):.3f}")
    print("Consistency Scores:")
    print("Min & Max & Mean & Std")
    print(f"{np.min(all_consistency_scores):.3f} & {np.max(all_consistency_scores):.3f} & {np.mean(all_consistency_scores):.3f} & {np.std(all_consistency_scores):.3f}")
    print("Overall Scores:")
    print("Min & Max & Mean & Std")
    print(f"{np.min(all_overall_scores):.3f} & {np.max(all_overall_scores):.3f} & {np.mean(all_overall_scores):.3f} & {np.std(all_overall_scores):.3f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)