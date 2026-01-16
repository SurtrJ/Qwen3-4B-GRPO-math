"""
Evaluate models with CONSISTENT system+user prompt format.

This script ensures all models receive the EXACT SAME input format:
- Full system prompt from the dataset
- User question

Usage:
    python scripts/evaluate_models_consistent.py --model base --num-samples 100
    python scripts/evaluate_models_consistent.py --model sft --num-samples 100
    python scripts/evaluate_models_consistent.py --model grpo --num-samples 100
"""

import argparse
import json
import re
import torch
from pathlib import Path
from datasets import load_from_disk
from unsloth import FastLanguageModel
from vllm import SamplingParams
from tqdm import tqdm
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a single model with consistent system+user prompts")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['base', 'sft', 'grpo'],
        help='Model to evaluate'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to evaluate'
    )
    return parser.parse_args()

def load_model(model_type: str, base_model_path: str, lora_path: str = None):
    """Load model for evaluation"""
    print(f"\nLoading {model_type.upper()} model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=2048,
        load_in_4bit=False,
        fast_inference=True,
        max_lora_rank=32,
        gpu_memory_utilization=0.5,
        local_files_only=True,
    )

    # Setup custom chat template (same as training)
    system_prompt = """You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>"""

    reasoning_start = "<start_working_out>"
    reasoning_end = "</end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    chat_template = \
        "{% if messages[0]['role'] == 'system' %}"\
            "{{ messages[0]['content'] + eos_token }}"\
            "{% set loop_messages = messages[1:] %}"\
        "{% else %}"\
            "{{ '{system_prompt}' + eos_token }}"\
            "{% set loop_messages = messages %}"\
        "{% endif %}"\
        "{% for message in loop_messages %}"\
            "{% if message['role'] == 'user' %}"\
                "{{ message['content'] }}"\
            "{% elif message['role'] == 'assistant' %}"\
                "{{ message['content'] + eos_token }}"\
            "{% endif %}"\
        "{% endfor %}"\
        "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
        "{% endif %}"

    chat_template = chat_template\
        .replace("'{system_prompt}'", f"'{system_prompt}'")\
        .replace("'{reasoning_start}'", f"'{reasoning_start}'")
    tokenizer.chat_template = chat_template

    # Load LoRA if specified
    if lora_path:
        print(f"  Loading LoRA from: {lora_path}")
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
            lora_alpha=64,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)

    print(f"✓ {model_type.upper()} model loaded")
    return model, tokenizer

def extract_messages_from_sample(sample):
    """
    Extract system and user messages from dataset sample.

    Handles both formats:
    1. list: [{system}, {user}]
    2. str: just the user question
    """

    prompt = sample['prompt']

    # Case 1: prompt is a list with system and user messages
    if isinstance(prompt, list):
        system_msg = None
        user_msg = None

        for msg in prompt:
            if msg.get('role') == 'system':
                system_msg = msg.get('content')
            elif msg.get('role') == 'user':
                user_msg = msg.get('content')

        # If system message is missing, use default
        if system_msg is None:
            system_msg = """You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>"""

        return system_msg, user_msg

    # Case 2: prompt is a string (just the user question)
    elif isinstance(prompt, str):
        # Use the same system prompt as training
        system_msg = """You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>"""
        user_msg = prompt
        return system_msg, user_msg

    # Case 3: unexpected format
    else:
        print(f"⚠️  Unexpected prompt format: {type(prompt)}")
        system_msg = """You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>"""
        user_msg = str(prompt)
        return system_msg, user_msg

def generate_response(model, tokenizer, system_prompt: str, user_question: str) -> str:
    """
    Generate model response using CONSISTENT system+user prompt format.

    This is the CRITICAL FIX: all models receive the same message format.
    """

    # Build messages with BOTH system and user prompts
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_k=50,
        max_tokens=2048,
    )

    output = model.fast_generate(
        [text],
        sampling_params=sampling_params,
        lora_request=None,
    )[0].outputs[0].text

    return output

def evaluate_response(response: str, ground_truth: str):
    """Calculate all metrics for a single response"""
    reasoning_end = "</end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    metrics = {}

    # Compile regex patterns
    match_format = re.compile(
        rf"{reasoning_end}.*?{solution_start}(.+?){solution_end}[\s]{{0,}}$",
        flags=re.MULTILINE | re.DOTALL
    )

    match_numbers = re.compile(
        solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
        flags=re.MULTILINE | re.DOTALL
    )

    # Format metrics
    metrics['format_exact'] = 1.0 if match_format.search(response) else 0.0

    format_approx = 0.0
    format_approx += 0.5 if response.count(reasoning_end) == 1 else -1.0
    format_approx += 0.5 if response.count(solution_start) == 1 else -1.0
    format_approx += 0.5 if response.count(solution_end) == 1 else -1.0
    metrics['format_approx'] = max(format_approx, -4)

    # Answer extraction
    extracted = None
    match = match_format.search(response)
    if match:
        extracted = match.group(1)

    extracted_number = None
    match_num = match_numbers.search(response)
    if match_num:
        extracted_number = match_num.group(1).replace(",", "")

    # Answer accuracy
    metrics['answer_exact'] = 0.0
    metrics['answer_numerical'] = 0.0

    if extracted:
        if extracted == ground_truth or extracted.strip() == ground_truth.strip():
            metrics['answer_exact'] = 1.0

    if extracted_number:
        try:
            pred_float = float(extracted_number)
            true_float = float(ground_truth.strip())

            if pred_float == true_float:
                metrics['answer_numerical'] = 1.0
                metrics['answer_exact'] = 1.0
            else:
                ratio = pred_float / true_float if true_float != 0 else 0
                if 0.9 <= ratio <= 1.1:
                    metrics['answer_numerical'] = 0.8
                elif 0.8 <= ratio <= 1.2:
                    metrics['answer_numerical'] = 0.6
        except (ValueError, ZeroDivisionError):
            pass

    # Composite score
    metrics['composite'] = (
        0.2 * metrics['format_exact'] +
        0.1 * ((metrics['format_approx'] + 4) / 6) +
        0.5 * metrics['answer_exact'] +
        0.2 * metrics['answer_numerical']
    )

    return metrics

def main():
    args = parse_args()

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    BASE_MODEL_PATH = "/root/autodl-tmp/unsloth/Qwen3-4B-Base"
    TEST_DATA_PATH = str(BASE_DIR / "data" / "test_split")
    OUTPUT_DIR = BASE_DIR / "evaluation_results_consistent"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"EVALUATING {args.model.upper()} MODEL (CONSISTENT SYSTEM+USER PROMPTS)")
    print("="*80)

    # Load test dataset
    print(f"\nLoading test dataset from: {TEST_DATA_PATH}")
    test_dataset = load_from_disk(TEST_DATA_PATH)
    print(f"✓ Loaded {len(test_dataset)} test samples")

    # Determine LoRA path
    lora_path = None
    if args.model == 'sft':
        lora_path = str(BASE_DIR / "checkpoints" / "sft_checkpoint")
    elif args.model == 'grpo':
        lora_path = str(BASE_DIR / "grpo_saved_lora")

    # Load model
    model, tokenizer = load_model(args.model, BASE_MODEL_PATH, lora_path)

    # Evaluate
    print(f"\nEvaluating on {args.num_samples} samples...")
    results = []

    for i in tqdm(range(args.num_samples), desc=f"Evaluating {args.model}"):
        sample = test_dataset[i]

        # CRITICAL FIX: Extract BOTH system and user messages from dataset
        system_prompt, user_question = extract_messages_from_sample(sample)
        ground_truth = sample['answer']

        # Generate response with CONSISTENT format
        response = generate_response(model, tokenizer, system_prompt, user_question)

        # Evaluate
        metrics = evaluate_response(response, ground_truth)

        # Save with full context for debugging
        results.append({
            'index': i,
            'system_prompt': system_prompt,  # Save system prompt for verification
            'user_question': user_question,  # Save user question
            'prompt': user_question,  # For backward compatibility
            'ground_truth': ground_truth,
            'response': response,
            **metrics
        })

    # Save results
    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / f'{args.model}_model_results.json'
    results_df.to_json(output_path, orient='records', indent=2)

    # Print summary
    print(f"\n{args.model.upper()} Model Summary:")
    print(f"  Format Exact: {results_df['format_exact'].mean():.3f}")
    print(f"  Format Approx: {results_df['format_approx'].mean():.3f}")
    print(f"  Answer Exact: {results_df['answer_exact'].mean():.3f}")
    print(f"  Answer Numerical: {results_df['answer_numerical'].mean():.3f}")
    print(f"  Composite Score: {results_df['composite'].mean():.3f}")

    # Verify system prompt consistency
    first_system = results[0]['system_prompt']
    all_same = all(r['system_prompt'] == first_system for r in results)
    print(f"\n✓ System prompt consistency: {'All same' if all_same else 'Different!'}")

    print(f"\n✓ Results saved to: {output_path}")

    # Cleanup
    print("\nCleaning up...")
    try:
        del model
        del tokenizer
    except:
        pass

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.synchronize()

    print("✓ Cleanup complete")

if __name__ == "__main__":
    main()
