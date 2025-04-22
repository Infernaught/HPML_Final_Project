import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI
import pandas as pd
import re
from training.constants import AIME_TRAIN_DATASET_PATH_DISTILLED
import datasets

client = OpenAI()

def get_answer_from_output(output: str):
    try:
        regex = r"<answer>\s*(\d+)\s*<\/answer>"
        match = re.search(regex, output)
        solution = match.group(1).strip()
        return solution
    except Exception:
        return None

def get_distilled_aime_dataset(df: pd.DataFrame, output_path: str):
    new_dataset = []
    for index, row in df.iterrows():
        question = row["Question"]
        answer = row["Answer"]
        prompt = [
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
        ]
        try:
            response = client.responses.create(model="o4-mini", input=prompt).output_text
        except Exception as e:
            print("Error: ", e)
            continue
        print("--------------------------------")
        print("Correct answer: ", answer)
        print("--------------------------------")
        print("Generated answer: ", response)
        print("--------------------------------")
        if get_answer_from_output(response) == answer:
            print("Answer is correct")
            new_dataset.append({"prompt": question, "completion": response})
        else:
            print("Answer is incorrect")
        print("--------------------------------")
        if len(new_dataset) % 50 == 0:
            new_df = pd.DataFrame(new_dataset)
            new_df.to_json(output_path, orient="records", lines=True)
    new_df = pd.DataFrame(new_dataset)
    new_df.to_json(output_path, orient="records", lines=True)

if __name__ == "__main__":
    dataset = datasets.load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    df = pd.DataFrame(dataset)

    # Remove rows where "2024" is not the year
    train_df = df[(df["Year"] != 2024)]

    get_distilled_aime_dataset(train_df, AIME_TRAIN_DATASET_PATH_DISTILLED)
