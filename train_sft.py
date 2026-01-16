"""
Train SFT (Supervised Fine-Tuning) model using OpenMathReasoning-mini dataset.
This creates the format pre-training checkpoint for comparison with Base and GRPO models.

Training parameters (from notebook):
- Dataset: OpenMathReasoning-mini (filtered to 59 samples)
- Epochs: 5
- Batch size: 2 (per device) x 4 (gradient accumulation) = 8 effective
- Learning rate: 5e-5
- LoRA rank: 32
- Max sequence length: 2048

Output: checkpoints/sft_checkpoint/
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
import pandas as pd
import numpy as np
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints" / "sft_checkpoint"
BASE_MODEL_PATH = "/root/autodl-tmp/unsloth/Qwen3-4B-Base"
DATASET_PATH = "/root/autodl-tmp/datasets/OpenMathReasoning-mini"

# Training hyperparameters (from notebook)
max_seq_length = 2048
lora_rank = 32

print("="*60)
print("SFT Training - Format Pre-training")
print("="*60)

# Custom format strings
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

print("\nLoading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.5,
    local_files_only=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank*2,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

print("✓ Model loaded")

# Setup custom chat template
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
print("✓ Custom chat template set")

# Load and format OpenMathReasoning-mini dataset
print(f"\nLoading dataset from: {DATASET_PATH}")
dataset = load_dataset(DATASET_PATH, split="cot")
dataset = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]

# Filter to numerical answers only
is_number = pd.to_numeric(pd.Series(dataset["expected_answer"]), errors="coerce").notnull()
dataset = dataset.iloc[np.where(is_number)[0]]
print(f"✓ Loaded {len(dataset)} samples (numerical answers only)")

# Format function
def format_dataset(x):
    expected_answer = x["expected_answer"]
    problem = x["problem"]

    # Remove existing reasoning tags
    thoughts = x["generated_solution"]
    thoughts = thoughts.replace("```", "").replace("```", "")
    thoughts = thoughts.strip()

    # Add custom formatting
    final_prompt = \
        reasoning_start + thoughts + reasoning_end + \
        solution_start + expected_answer + solution_end

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": final_prompt},
    ]

dataset["Messages"] = dataset.apply(format_dataset, axis=1)

# Filter by length (max_seq_length/2 = 1024 tokens)
dataset["N"] = dataset["Messages"].apply(lambda x: len(tokenizer.apply_chat_template(x)))
dataset = dataset.loc[dataset["N"] <= max_seq_length/2].copy()

print(f"✓ Filtered to {len(dataset)} samples (length <= {max_seq_length//2} tokens)")

# Convert to Hugging Face dataset
dataset["text"] = tokenizer.apply_chat_template(dataset["Messages"].values.tolist(), tokenize=False)
dataset = Dataset.from_pandas(dataset)

# Setup trainer
print("\nSetting up SFT trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=5,
        learning_rate=5e-5,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    ),
)

print("✓ Trainer configured")
print(f"\nTraining configuration:")
print(f"  - Samples: {len(dataset)}")
print(f"  - Epochs: 5")
print(f"  - Batch size: 2 x 4 GA = 8 effective")
print(f"  - Learning rate: 5e-5")
print(f"  - Max sequence length: {max_seq_length}")

# Train
print("\n" + "="*60)
print("Starting training...")
print("="*60)

trainer.train()

print("\n" + "="*60)
print("Training completed!")
print("="*60)

# Save checkpoint
print(f"\nSaving SFT checkpoint to: {CHECKPOINT_DIR}")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

model.save_lora(str(CHECKPOINT_DIR))
tokenizer.save_pretrained(str(CHECKPOINT_DIR))

print("✓ SFT checkpoint saved successfully")

# Cleanup
del dataset
torch.cuda.empty_cache()
import gc
gc.collect()

print("\n" + "="*60)
print("SFT training completed successfully!")
print("="*60)
