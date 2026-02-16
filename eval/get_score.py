import json
import argparse
from utils import *
from tqdm import tqdm
import numpy as np

task_list = [
"scenario_understanding", "causality_discovery", "event_aware_forecasting", "decision_making"
]


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument('--input_path', type=str, default="data/test_id.json", help="Path of the answer file")
    parser.add_argument('--output_path', type=str, default="data/res_test_id.json", help="Path of the output file")
    parser.add_argument('--proc_total', type=int, default=8, help="Process total numbers.")
    parser.add_argument('--model_name', type=str, default="your_model_name", help="Process id.")
    parser.add_argument('--task_type', type=str, default=None, 
                       help="Task types to evaluate. --task_type causality_discovery, event_aware_forecasting")
    return parser.parse_args()


def detect_and_load_json(file_path):
    with open(file_path, "r", encoding='utf-8') as F:
        content = F.read().strip()
    
    # Try to parse as JSON array first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    
    # If not a JSON array, try line-by-line parsing
    test_data = []
    with open(file_path, "r", encoding='utf-8') as F:
        for line in F:
            if line.strip():  # Skip empty lines
                try:
                    test_data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    continue
    
    return test_data


if __name__ == "__main__":
    args = get_args()
    test_data = []
    if args.proc_total == 0:
        test_data = detect_and_load_json(args.input_path)
    else:
        for proc_id in range(args.proc_total):
            file_path = f"{args.input_path[:-5]}{proc_id}.json"
            test_data.extend(detect_and_load_json(file_path))

    # Filter test data based on task_type if specified
    if args.task_type:
        selected_tasks = [t.strip() for t in args.task_type.split(',')]
        test_data = [item for item in test_data if item["task_type"] in selected_tasks]
        task_list = selected_tasks

    # Stats data
    stats_data = {task: {"rewards": [], "valid_flags": [], "valid_rewards": []} for task in task_list}
    
    for item in tqdm(test_data, total=len(test_data)):
        task_type = item["task_type"]
        # Use pred_response if pred_rat is not available
        pred_content = item.get("pred_rat", item.get("pred_response", ""))
        
        # If pred_content is null, None, or empty, return reward 0 and invalid
        if pred_content is None or pred_content == "" or pred_content == "null":
            reward = 0.0
            is_valid = False
        else:
            reward, is_valid = compute_reward(pred_content, item["response"], item["problem"], task_type, args.model_name)
            reward = float(reward)
        
        # Record stats data
        stats_data[task_type]["rewards"].append(reward)
        stats_data[task_type]["valid_flags"].append(is_valid)
        if is_valid:
            stats_data[task_type]["valid_rewards"].append(reward)
        
        # Save to item
        item["reward"] = reward
        item["is_valid"] = is_valid

    # Calculate three metrics
    final_stats = {}
    for task in task_list:
        rewards = stats_data[task]["rewards"]
        valid_flags = stats_data[task]["valid_flags"]
        valid_rewards = stats_data[task]["valid_rewards"]
        
        total_count = len(rewards)
        valid_count = sum(valid_flags)
        
        # Overall accuracy (for discrete-output tasks are accuracy, for sequence-output tasks are average MAE)
        if task in ["scenario_understanding", "causality_discovery", "decision_making"]:
            overall_accuracy = np.mean(rewards) if total_count > 0 else 0.0
            valid_accuracy = np.mean(valid_rewards) if valid_count > 0 else 0.0
        else:  # Event_aware_forecasting
            overall_accuracy = np.mean(rewards) if total_count > 0 else 0.0
            valid_accuracy = np.mean(valid_rewards) if valid_count > 0 else 0.0
        
        # Success rate (legal rate)
        success_rate = valid_count / total_count if total_count > 0 else 0.0
        
        final_stats[task] = {
            "overall_score": overall_accuracy,
            "success_rate": success_rate, 
            "valid_score": valid_accuracy,
            "total_samples": total_count,
            "valid_samples": valid_count
        }
    
    # Calculate overall stats
    all_rewards = []
    all_valid_flags = []
    all_valid_rewards = []
    for task in task_list:
        all_rewards.extend(stats_data[task]["rewards"])
        all_valid_flags.extend(stats_data[task]["valid_flags"])
        all_valid_rewards.extend(stats_data[task]["valid_rewards"])
    
    total_samples = len(all_rewards)
    total_valid = sum(all_valid_flags)
    
    final_stats["overall"] = {
        "overall_score": np.mean(all_rewards) if total_samples > 0 else 0.0,
        "success_rate": total_valid / total_samples if total_samples > 0 else 0.0,
        "valid_score": np.mean(all_valid_rewards) if total_valid > 0 else 0.0,
        "total_samples": total_samples,
        "valid_samples": total_valid
    }

    final_output = [final_stats] + test_data
    with open(args.output_path, "w", encoding='utf-8') as F:
        json.dump(final_output, F, indent=2, ensure_ascii=False)