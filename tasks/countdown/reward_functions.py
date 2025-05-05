def format_reward_func(prompts, completions, nums, target):
    import re

    rewards = []
    for prompt, completion, n, t in zip(prompts, completions, nums, target):
        reward = 0
        try:
            completion = "<think>" + completion
            regex = (
                    r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\n"
                    r"<answer>\s*([\s\S]*?)\s*<\/answer>$"
                )
            match = re.search(regex, completion, re.DOTALL)
            if match is not None and len(match.groups()) == 2:
                    reward = 1.0
        except Exception:
            pass
        print(f"Format reward: {reward}")
        rewards.append(reward)
    return rewards


# Check if the output contains the correct answer
def equation_reward_func(prompts, completions, nums, target):
    import re
    import ast

    rewards = []
    for prompt, completion, ind_nums, ind_target in zip(prompts, completions, nums, target):
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print(f"Ind nums: {ind_nums}")
        print(f"Ind target: {ind_target}")
        reward = 0
        try:
            # add synthetic <think> as its already part of the prompt and prefilled
            # for the assistant to more easily match the regex
            completion = "<think>" + completion
            match = re.search(r"<answer>\s*([\s\S]*?)\s*<\/answer>", completion)
            if not match:
                print("No answer found in completion. Equation reward: 0.0")
                rewards.append(0.0)
                continue

            # Extract the "answer" part from the completion
            equation = match.group(1).strip()
            # Extract all numbers from the equation
            used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

            # Convert the example["nums"] to a list if it's a string
            # This is common for columns like lists in datasets
            if isinstance(ind_nums, str):
                ind_nums = ast.literal_eval(ind_nums)

            # Check if all numbers are used exactly once
            if sorted(used_numbers) != sorted(ind_nums):
                print("Numbers used in equation not the same as in example. Equation reward: 0.0")
                rewards.append(0.0)
                continue

            # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
            allowed_pattern = r'^[\d+\-*/().\s]+$'
            if not re.match(allowed_pattern, equation):
                print("Equation contains invalid characters. Equation reward: 0.0")
                rewards.append(0.0)
                continue

            # Evaluate the equation with restricted globals and locals
            result = eval(equation, {"__builtins__": None}, {})
            # Check if the equation is correct and matches the ground truth
            if abs(float(result) - float(ind_target)) < 1e-5:
                rewards.append(1.0)
                reward = 1.0
            else:
                print("Equation is incorrect. Equation reward: 0.0")
                rewards.append(0.0)
            
            print(f"Equation reward: {reward}")
            print("--------------------------------")

        except Exception:
            pass

    return rewards