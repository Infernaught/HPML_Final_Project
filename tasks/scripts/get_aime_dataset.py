import datasets
import pandas as pd
from transformers import AutoTokenizer

dataset = datasets.load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
df = pd.DataFrame(dataset)

# Remove rows where "2024" is not the year
train_df = df[df["Year"] != 2024]
eval_df = df[df["Year"] == 2024]

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

def format_row(row):
    question = row["Question"]
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a helpful assistant. You first think about the reasoning process step by step "
            "and then provide the user with an answer.",
        },
        {
            "role": "user",
            "content": (
                f"This is a question from the AIME: \n {question} \nPlease answer the question. "
                "Show your work in <think> </think> tags. And return the final answer in <answer> </answer> "
                "tags. For example: <answer> 33 </answer>."
            ),
        },
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>",
        },
    ]
    return {
        "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True),
        "answer": row["Answer"],
    }

train_df = train_df.apply(format_row, axis=1)
train_df.to_json("./datasets/aime_train_dataset.jsonl", orient="records", lines=True)

eval_df = eval_df.apply(format_row, axis=1)
eval_df.to_json("./datasets/aime_eval_dataset.jsonl", orient="records", lines=True)