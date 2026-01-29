# HF instruct-dataset: primeai7460/hackmentor-instruction, 13,982 samples -> 0.5-1% improvment
# Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset, 53,201 samples
# AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.0, 83,920 samples

from datasets import load_dataset

def format_instruct1(example):
    return {"messages": [
        #{"role": "system", "content": 'you are a helpful cybersecurity expert'},
        {"role": "user", "content": example["user"]}, # instruct-tuning: user
        {"role": "assistant", "content": example["assistant"]}, # instruct-tuning: assistant
        ]
    }

def format_instruct2(example):
    return {
        "messages": [
        {"role": "system", "content": 'you are a helpful cybersecurity expert'},
        {"role": "user", "content": example["user"]}, # instruct-tuning: user
        {"role": "assistant", "content": example["assistant"]}, # instruct-tuning: assistant
        ]
    }

def format_instruct3(example):
    return {
        "messages": [
        {"role": "system", "content": 'You are an expert cybersecurity assistant. You provide accurate, detailed, and safe information about cybersecurity threats, vulnerabilities, and mitigations.'},
        {"role": "user", "content": example["user"]}, # instruct-tuning: user
        {"role": "assistant", "content": example["assistant"]}, # instruct-tuning: assistant
        ]
    }

# mapping
def format_data(dataset: str):
    
    data = load_dataset(dataset, split="train")

    formatted_data = data.map(format_instruct1, remove_columns=data.column_names)

    return formatted_data # for training
    #return formatted_data['messages'] # for printing

# for testing:
#formatted_dataset = format_data('Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset')
#print(formatted_dataset[0])
