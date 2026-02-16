import os
import math
import json
import argparse
import itertools
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from vllm import LLM, SamplingParams
from statistics import median
import random

def set_prompt(question: str, system_prompt: str):
    return_prompt = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n")
    return return_prompt


def get_message(data_item):
    message_item = [{
        "question_id": data_item['question_id'],
        "prompt": set_prompt(data_item['problem'], data_item['system']),
    }]
    return message_item


def batched_iterable(iterable, batch_size):
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, batch_size))
        if not batch:
            break
        yield batch


def main(args):
    torch.cuda.empty_cache()
    with open(args.test_file, "r", encoding='utf-8') as F:
        test_data = json.load(F)
    random.seed(42)
    random.shuffle(test_data)
    # Filter test data based on task_type if specified
    if args.task_type:
        selected_tasks = [t.strip() for t in args.task_type.split(',')]
        test_data = [item for item in test_data if item['task_type'] in selected_tasks]
        print(f"Filtered {len(test_data)} samples for tasks: {', '.join(selected_tasks)}")

    unit = int(math.ceil(len(test_data)/args.proc_total))
    print(args.proc_id, len(test_data), unit*args.proc_id, min(unit*(args.proc_id+1), len(test_data)))
    test_data = test_data[unit*args.proc_id: min(unit*(args.proc_id+1), len(test_data))]

    if os.path.isfile(f"{args.output_path[:-5]}{args.proc_id}.json"):
        with open(f"{args.output_path[:-5]}{args.proc_id}.json", "r", encoding='utf-8') as F:
            cur_test_length = sum(1 for _ in F)
        print(cur_test_length)
        test_data = test_data[cur_test_length:]
    else:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
    id_info_mapping = {}
    for item in test_data:
        domain = item.get("domain", "")
        id_info_mapping[item['question_id']] = {
            "question_id": item['question_id'],
            "problem": item['problem'],
            "response": item['response'],
            "task_type": item['task_type'],
            "domain": domain,
            "system": item['system'],
        }

    llm = LLM(
        model=args.model_dir,
        max_model_len=8192,
        max_num_seqs=args.batch_size,
        tensor_parallel_size=args.parallel_size,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=args.max_model_len,
        stop_token_ids=[],
    )


    res_data = {}
    token_lens = []          


    for batch in tqdm(batched_iterable(test_data, args.batch_size), total=int(math.ceil(len(test_data)/args.batch_size)), desc=f"{args.proc_id}_batch_infer"):
        messages = []
        token_lens_in_batch = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(get_message, item): item for item in batch}
            for future in as_completed(futures):
                messages.extend(future.result())
        responses = llm.generate(messages, sampling_params=sampling_params, use_tqdm=False)
        
        for msg, res in zip(messages, responses):
            generated_text = res.outputs[0].text

            # token number
            tok_len = len(llm.get_tokenizer().encode(generated_text, add_special_tokens=False))
            token_lens.append(tok_len)
            token_lens_in_batch.append(tok_len)

            cur_item = {
                "question_id": msg['question_id'],
                "problem": id_info_mapping[msg['question_id']]['problem'],
                "pred_rat": generated_text,
                "response": id_info_mapping[msg['question_id']]['response'],
                "task_type": id_info_mapping[msg['question_id']]['task_type'],
                "domain": id_info_mapping[msg['question_id']]['domain'],
                "system": id_info_mapping[msg['question_id']]['system'],
            }
            with open(f"{args.output_path[:-5]}{args.proc_id}.json", "a", encoding='utf-8') as F:
                F.write(f"{json.dumps(cur_item, ensure_ascii=False)}\n")

            # token related statistics after one batch is generated
            if token_lens_in_batch:                      # avoid division by 0
                avg_len = sum(token_lens_in_batch) / len(token_lens_in_batch)
                min_len = min(token_lens_in_batch)
                max_len = max(token_lens_in_batch)
                med_len = median(token_lens_in_batch)

                print(f"\nToken length stats in one batch (proc {args.proc_id}): "
                    f"avg={avg_len:.1f}, min={min_len}, max={max_len}, median={med_len}")
            else:
                print("No generations to report token statistics.")

    # token related statistics after all samples are generated
    if token_lens: # avoid division by 0
        avg_len = sum(token_lens) / len(token_lens)
        min_len = min(token_lens)
        max_len = max(token_lens)
        med_len = median(token_lens)

        print(f"\nToken length stats (proc {args.proc_id}): "
            f"avg={avg_len:.1f}, min={min_len}, max={max_len}, median={med_len}")
    else:
        print("No generations to report token statistics.")


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Inference using customized model.")
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the pretrained model directory.")
    parser.add_argument('--test_file', type=str, required=True, help="Path to the test data file in JSON format.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the output answers.")
    parser.add_argument('--proc_total', type=int, required=True, help="Process total numbers.")
    parser.add_argument('--proc_id', type=int, required=True, help="Process id.")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size for processing.")
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--parallel_size', type=int, default=1)
    parser.add_argument('--max_model_len', type=int, default=4096, help="Max model length.")
    parser.add_argument('--task_type', type=str, default=None, 
                       help="Task types to evaluate. --task_type causality_discovery, event_aware_forecasting")
    args = parser.parse_args()

    # Run main function with provided arguments
    main(args)
