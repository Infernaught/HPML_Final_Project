import datasets
import pandas as pd
from transformers import AutoTokenizer
import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from training.constants import AVAILABLE_MODELS

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate Countdown dataset with formatted prompts')
parser.add_argument('--base_model', type=str, required=True, 
                    help='Base model to use for tokenization and prompt formatting',
                    choices=AVAILABLE_MODELS.keys())
parser.add_argument('--output_dir', type=str, default='./tasks/countdown',
                    help='Directory to save the output files (default: ./tasks/countdown)')
parser.add_argument('--suffix', type=str, default='',
                    help='Suffix to add to the output file names (default: "")')
args = parser.parse_args()

# Validate the base model
if args.base_model not in AVAILABLE_MODELS:
    print(f"Error: {args.base_model} is not in the list of available models.")
    print(f"Available models: {', '.join(AVAILABLE_MODELS)}")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

train_dataset = datasets.load_dataset("predibase/countdown", split="train")
eval_dataset = datasets.load_dataset("predibase/countdown", split="test")

train_df = pd.DataFrame(train_dataset)
eval_df = pd.DataFrame(eval_dataset)

# Remove rows where "2024" is not the year
train_df = train_df[:100]
eval_df = eval_df[:20]

tokenizer = AutoTokenizer.from_pretrained(AVAILABLE_MODELS[args.base_model])

def format_row(row):
    nums = row["nums"]
    target = row["target"]
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process step by step and then provide the user with an answer.",
        },
        {
            "role": "user",
            "content": (
                f"Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and parentheses, and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
            ),
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>",
        },
    ]
    return {
        "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
        "nums": nums,
        "target": target,
    }

train_df = train_df.apply(format_row, axis=1)
train_output_path = os.path.join(args.output_dir, f"countdown_train_dataset{args.suffix}.jsonl")
train_df.to_json(train_output_path, orient="records", lines=True)
print(f"Saved train dataset to {train_output_path}")

eval_df = eval_df.apply(format_row, axis=1)
eval_output_path = os.path.join(args.output_dir, f"countdown_eval_dataset{args.suffix}.jsonl")
eval_df.to_json(eval_output_path, orient="records", lines=True)
print(f"Saved eval dataset to {eval_output_path}")