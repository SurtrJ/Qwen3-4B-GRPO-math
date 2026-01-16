"""
Create train/val/test split from DAPO-Math-17k dataset.
This ensures a held-out test set for fair model comparison.

Output:
- Train set: 80% (for GRPO training)
- Validation set: 10% (for GRPO monitoring)
- Test set: 10% (held out for final evaluation)
"""

import json
from datasets import load_dataset
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET_PATH = "/root/autodl-tmp/datasets/DAPO-Math-17k-Processed"

print("="*60)
print("Creating train/val/test split from DAPO-Math-17k")
print("="*60)

# Load dataset
print("\nLoading dataset from:", DATASET_PATH)
dataset = load_dataset(DATASET_PATH, "en", split="train")
print(f"Loaded {len(dataset)} samples")

# Process dataset to add 'answer' field (same as notebook cell 37)
def extract_hash_answer(text):
    return text

dataset = dataset.map(lambda x: {
    "prompt": [
        {"role": "system", "content": "You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>"},
        {"role": "user", "content": x["prompt"]},
    ],
    "answer": extract_hash_answer(x["solution"]),
})
print(f"✓ Dataset processed with 'prompt' and 'answer' fields")

# Create 80/10/10 split using fixed seed for reproducibility
print("\nCreating train/val/test split (80/10/10)...")

# First split: 80% train+val, 20% test
split_1 = dataset.train_test_split(test_size=0.2, seed=3407)
train_val = split_1["train"]
test_set = split_1["test"]

# Second split: 80% train, 10% val (from the 80% train+val)
# We need 0.125 of the 80% to get 10% of total
split_2 = train_val.train_test_split(test_size=0.125, seed=3407)
train_set = split_2["train"]
val_set = split_2["test"]

print(f"Train set: {len(train_set)} samples ({len(train_set)/len(dataset)*100:.1f}%)")
print(f"Validation set: {len(val_set)} samples ({len(val_set)/len(dataset)*100:.1f}%)")
print(f"Test set: {len(test_set)} samples ({len(test_set)/len(dataset)*100:.1f}%)")

# Create data directory
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Save splits
print("\nSaving splits to disk...")
train_path = DATA_DIR / "train_split"
val_path = DATA_DIR / "val_split"
test_path = DATA_DIR / "test_split"

train_set.save_to_disk(str(train_path))
val_set.save_to_disk(str(val_path))
test_set.save_to_disk(str(test_path))

print(f"✓ Train set saved to: {train_path}")
print(f"✓ Validation set saved to: {val_path}")
print(f"✓ Test set saved to: {test_path}")

# Save split metadata for verification
metadata = {
    "total_samples": len(dataset),
    "train_samples": len(train_set),
    "val_samples": len(val_set),
    "test_samples": len(test_set),
    "train_percentage": len(train_set)/len(dataset)*100,
    "val_percentage": len(val_set)/len(dataset)*100,
    "test_percentage": len(test_set)/len(dataset)*100,
    "seed": 3407,
    "dataset_path": DATASET_PATH,
}

metadata_path = DATA_DIR / "split_metadata.json"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n✓ Metadata saved to: {metadata_path}")

# Save test set indices for verification
test_indices = {"test_indices": test_set['extra_info'][:100]}  # Save first 100 for reference
test_indices_path = DATA_DIR / "test_indices_sample.json"
with open(test_indices_path, 'w') as f:
    json.dump(test_indices, f, indent=2)

print(f"✓ Test indices sample saved to: {test_indices_path}")

print("\n" + "="*60)
print("Dataset split creation completed!")
print("="*60)
