import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from openai import OpenAI
import pandas as pd
import re
import datasets

client = OpenAI()

def check_equation(completion, nums, target):
    import re
    import ast

    try:
        # add synthetic <think> as its already part of the prompt and prefilled
        # for the assistant to more easily match the regex
        completion = "<think>" + completion
        match = re.search(r"<answer>\s*([\s\S]*?)\s*<\/answer>", completion)
        if not match:
            print("No answer found in completion. Equation reward: 0.0")
            return False

        # Extract the "answer" part from the completion
        equation = match.group(1).strip()
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

        # Convert the example["nums"] to a list if it's a string
        # This is common for columns like lists in datasets
        if isinstance(nums, str):
            nums = ast.literal_eval(nums)

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(nums):
            print("Numbers used in equation not the same as in example. Equation reward: 0.0")
            return False

        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation):
            print("Equation contains invalid characters. Equation reward: 0.0")
            return False

        # Evaluate the equation with restricted globals and locals
        result = eval(equation, {"__builtins__": None}, {})
        # Check if the equation is correct and matches the ground truth
        if abs(float(result) - float(target)) < 1e-5:
            return True
        else:
            print("Equation is incorrect.")
            return False

    except Exception:
        return False


def get_distilled_aime_dataset(df: pd.DataFrame, output_path: str):
    new_dataset = []
    for index, row in df.iterrows():
        nums = row["nums"]
        target = row["target"]
        prompt = [
            {
                "role": "system",
                "content": "You are a helpful assistant. You first think about the reasoning process step by step "
                "and then provide the user with an answer.",
            },
            {
                "role": "user",
                "content": (
                    f"Using the numbers {nums}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and parentheses, and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>. As shown in the example, please omit the equals  sign and the right hand side of the equation."
                ),
            },
        ]
        try:
            response = client.responses.create(model="o4-mini", input=prompt).output_text
        except Exception as e:
            print("Error: ", e)
            continue
        print("--------------------------------")
        print("Generated answer: ", response)
        print("--------------------------------")
        if check_equation(response, nums, target):
            print("Answer is correct")
            new_dataset.append({"nums": nums, "target": target, "completion": response})
        else:
            print("Answer is incorrect")
        print("--------------------------------")
        if len(new_dataset) % 50 == 0:
            new_df = pd.DataFrame(new_dataset)
            new_df.to_json(output_path, orient="records", lines=True)
    new_df = pd.DataFrame(new_dataset)
    new_df.to_json(output_path, orient="records", lines=True)

if __name__ == "__main__":
    dataset = datasets.load_dataset("predibase/countdown", split="train")
    df = pd.DataFrame(dataset)

    train_df = df[:300]

    get_distilled_aime_dataset(train_df, "tasks/countdown/countdown_train_dataset_distilled.jsonl")
