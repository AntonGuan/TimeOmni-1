import re
import ast
import numpy as np

def invalid_mae_penalty(gt):
    gt = np.array(gt, dtype=float)   # Ground truth sequence, like [1.2, 3.4, 2.1]
    zero_pred = np.zeros_like(gt)              # Generate a sequence with the same shape but all 0s [0.0, 0.0, 0.0]
    return float(np.mean(np.abs(zero_pred - gt)))


def forecasting_series_reward(pred, gt): 
    """ calculate MAE reward, output value ∈ (0, 1], can be used as GRPO reward. """ 
    alpha = 0.1 # or 0.05, not used in evaluation
    pred = np.array(pred, dtype=float) 
    gt = np.array(gt, dtype=float) 
    
    # Mean Absolute Error 
    mae = np.mean(np.abs(pred - gt)) 
    
    # Reward in (0,1] 
    reward = float(np.exp(-alpha * mae)) 
    return reward, mae

def extract_answer(text):
    """extract the content between <answer> and </answer> tags"""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def extract_list(text):
    """Safely extract the content inside [ and ] brackets"""
    match = re.search(r'\[(.*?)\]', text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return None

def compute_reward(pred_rat, response, problem, task_type, model_name):
    """
    calculate reward value
    return: (reward, is_valid) 
    - reward: 0.0/1.0 reward value, or MAE value
    - is_valid: boolean value, indicates whether the answer is valid
    """
    
    if task_type in ["scenario_understanding", "causality_discovery", "decision_making"]:
        content_answer = extract_answer(pred_rat) # Extract the output classification result
        if content_answer:
            # Remove any characters that are not letters or spaces
            content_answer = re.sub(r'[^a-zA-Z\s]', '', content_answer)
        solution_answer = response  # Extract the ground truth classification result
        if content_answer:
            # If the answer is successfully extracted, compare it
            reward = 1.0 if content_answer.strip().upper() == solution_answer.strip().upper() else 0.0
            is_valid = True  # Considered valid
        else:
            # If the answer is not successfully extracted, give 0
            reward = 0.0
            is_valid = False  # Not valid
        return reward, is_valid
    
    elif task_type == "event_aware_forecasting":
        # First parse ground truth
        gt_list = response
        if isinstance(gt_list, str):
            gt_list = ast.literal_eval(gt_list)
        if not isinstance(gt_list, (list, tuple)):
            raise ValueError("groundtruth is not a time series")
        
        # Try to parse prediction result
        try:
            content_answer = extract_list(extract_answer(pred_rat)) # Extract prediction sequence
            pred_list = ast.literal_eval(content_answer)
            if not isinstance(pred_list, (list, tuple)):
                pred_list = [pred_list]
        except Exception as e:
            # Prediction result parsing failed, not valid
            return invalid_mae_penalty(gt_list), False

        # Verify the two lists have same length
        if len(pred_list) == len(gt_list) and len(pred_list) > 0 and len(gt_list) > 0:
            # Convert elements to float
            try:
                pred = [float(x) for x in pred_list]
                gt   = [float(x) for x in gt_list]
            except Exception as e:
                return invalid_mae_penalty(gt_list), False
            # Use the MAE to evaluate the prediction result
            _, mae = forecasting_series_reward(pred, gt)
            return mae, True
        else:
            return invalid_mae_penalty(gt_list), False

    else:
        raise ValueError("task type({}): not supported".format(task_type))