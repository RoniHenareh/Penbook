# data/format_hackmentor.py

from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# Dataset column mappings
DATASET_COLUMNS = {
    "primeai7460/hackmentor-instruction": ("input", "output"),
    "Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset": ("user", "assistant"),
    "AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0": ("user", "assistant"),
}

# Instruct
def format_instruct1(example, user_col, assistant_col):
    messages = [
        {"role": "user", "content": example[user_col]},
        {"role": "assistant", "content": example[assistant_col]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# InstructSystem
def format_instruct2(example, user_col, assistant_col):
    messages = [
        {"role": "system", "content": "You are an expert cybersecurity assistant. "
        "You provide accurate, detailed, and safe information about cybersecurity threats, vulnerabilities, and mitigations."},
        {"role": "user", "content": example[user_col]},
        {"role": "assistant", "content": example[assistant_col]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

def format_data(use_system_prompt: bool = False):
    """
    If False, uses format_instruct1 (without system prompt)
    If True, uses format_instruct2 (with system prompt)
    """
    formatted_datasets = []
    formatter_fn = format_instruct2 if use_system_prompt else format_instruct1
    
    for dataset_name, (user_col, assistant_col) in DATASET_COLUMNS.items():
        #print(f"Loading {dataset_name}...")
        ds = load_dataset(dataset_name, split="train")
        formatted = ds.map(
            lambda x: formatter_fn(x, user_col, assistant_col),
            remove_columns=ds.column_names
        )
        formatted_datasets.append(formatted)
        #print(f"  ✓ {len(formatted):,} samples")
    
    combined_dataset = concatenate_datasets(formatted_datasets)
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    #print(f"\nTotal: {len(combined_dataset):,} samples")
    return combined_dataset

# For testing:

# Without system prompt
#dataset = format_data(use_system_prompt=False)

# no system prompt
dataset = format_data()
print(dataset[0]["text"][:500])
