import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_SYSTEM_PROMPT = (
    "Output Format:\n"
    "<think>Your step-by-step reasoning process that justifies your answer</think>\n"
    "<answer>Your final answer(Note: Only output a single uppercase letter of the correct option)</answer>"
)
DEFAULT_MODEL_DIR = "anton-hugging/TimeOmni-1-7B"


def build_messages(question: str, system_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def build_legacy_prompt(question: str, system_prompt: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in dtype_mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return dtype_mapping[dtype_name]


def load_model(model_dir: str, dtype, device_map: str, trust_remote_code: bool):
    """Load a TimeOmni-1 checkpoint across base architectures.

    TimeOmni-1-7B is a text-only Qwen2.5 model loaded by ``AutoModelForCausalLM``.
    TimeOmni-1-4B/9B are Qwen3.5 checkpoints whose declared architecture is
    ``Qwen3_5ForConditionalGeneration`` (a VLM). ``AutoModelForCausalLM`` still
    resolves these to the text-only causal head and skips the vision tower, which
    is what we want for time-series reasoning. We fall back to the image-text
    class only if the causal head is unavailable for a given checkpoint.
    """
    kwargs = dict(dtype=dtype, device_map=device_map, trust_remote_code=trust_remote_code)
    try:
        return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_dir, **kwargs)


def load_tokenizer(model_dir: str, trust_remote_code: bool):
    """Load the tokenizer.

    Both Qwen2.5 (7B) and Qwen3.5 (4B/9B) checkpoints load with the slow
    tokenizer; we fall back to the fast tokenizer only if the slow build is
    unavailable for a given checkpoint.
    """
    try:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code, use_fast=False)
    except Exception:
        return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code, use_fast=True)


def prepare_inputs(tokenizer, question: str, system_prompt: str):
    messages = build_messages(question, system_prompt)
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = build_legacy_prompt(question, system_prompt)
    return tokenizer(prompt, return_tensors="pt")


def main():
    parser = argparse.ArgumentParser(description="Single-prompt inference")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--question", type=str, default="You are given two time series related to river discharge measurements, expressed in m^3/s. Through causal discovery methods, we aim to identify potential causal relationships between different measuring stations from time-series data alone. The time series of J96A is: [3.35, 2.92, 2.61, 2.92, 4.48, 7, 15.71, 10.65, 7.16, 5.79, 5.42, 5.31, 5, 4.38, 3.87, 3.52, 3.21, 2.92, 2.51, 2.39, 2.21, 2.08, 1.9, 1.75, 1.62, 1.56, 1.43, 1.4, 1.31, 1.24, 1.24, 1.25, 0.96, 2.75, 2.54, 2.03, 2.27, 2.36, 2.24, 2, 2.36, 2.16, 2.4, 2.11, 2.04, 1.96, 2.35, 2.26, 2.45, 2.19, 2.15, 1.91, 1.8, 1.64, 1.53, 1.44], The time series of UC1U is [5.19, 4.52, 4.2, 4.45, 6.29, 8, 22, 12.66, 8.48, 7.51, 7.15, 7.24, 7.42, 6.85, 6.24, 5.75, 5.37, 4.84, 4.45, 4.24, 3.94, 3.72, 3.57, 3.29, 3.12, 3, 2.85, 2.74, 2.68, 2.59, 2.56, 2.49, 2.55, 4.37, 4.71, 3.58, 3.84, 3.88, 3.8, 3.44, 3.64, 3.46, 3.57, 3.34, 3.19, 3.03, 3.39, 3.24, 3.45, 3.19, 3.14, 3, 2.86, 2.66, 2.56, 2.47]. Please identify the causal relationships between the two measurement stations? The data is collected every 12 hours from 2021-02-01 to 2021-02-28 totally 56 points each series.\n\n\nOptions:\nA. UC1U is the cause and J96A is the effect.\nB. J96A is the cause and UC1U is the effect.\nC. J96A and UC1U are not causal.")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.001)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto", help='Use "auto" for Accelerate placement or provide a custom device map string.')
    parser.add_argument("--trust_remote_code", action="store_true", help="Enable trust_remote_code for model/tokenizer loading.")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_dir, args.trust_remote_code)
    model = load_model(
        args.model_dir,
        dtype=resolve_torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    inputs = prepare_inputs(tokenizer, args.question, args.system_prompt)
    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_length = inputs["input_ids"].shape[-1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    completion = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    print(completion)


if __name__ == "__main__":
    main()
