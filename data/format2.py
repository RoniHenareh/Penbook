# HF instruct-dataset: primeai7460/hackmentor-instruction, 13,982 samples -> 0.5-1% improvment
# Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset, 53,201 samples
# AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0, 83,920 samples

from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def format_instruct1(example):
    messages = [
        #{"role": "system", "content": "You are an expert cybersecurity assistant. You provide accurate, detailed, and safe information about cybersecurity threats, vulnerabilities, and mitigations."},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

def format_instruct2(example):
    messages = [
        {"role": "system", "content": "You are an expert cybersecurity assistant. You provide accurate, detailed, and safe information about cybersecurity threats, vulnerabilities, and mitigations."},
        {"role": "user", "content": example["user"]},
        {"role": "assistant", "content": example["assistant"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# mapping
def format_data(dataset_name: str):

    data = load_dataset(dataset_name, split="train")
    formatted_data = data.map(format_instruct1, remove_columns=data.column_names)

    return formatted_data

# for testing:
#formatted_dataset = format_data('Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset')
#print(formatted_dataset[0])
