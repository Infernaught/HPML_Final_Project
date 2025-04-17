def format_reward_func(prompts, completions, answer):
    import re

    rewards = []
    for prompt, completion, a in zip(prompts, completions, answer):
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

def returns_int_reward_func(prompts, completions, answer):
    import re

    rewards = []
    for prompt, completion, a in zip(prompts, completions, answer):
        completion = "<think>" + completion
        reward = 0
        try:
            regex = r"<answer>\s*(\d+)\s*<\/answer>"
            match = re.search(regex, completion)
            if match is not None and len(match.groups()) == 1:
                reward = 1.0
        except Exception:
            pass
        print(f"Returns int reward: {reward}")
        rewards.append(reward)

    return rewards

def equation_reward_func(prompts, completions, answer):
    import re

    rewards = []
    for prompt, completion, a in zip(prompts, completions, answer):
        completion = "<think>" + completion
        reward = 0
        try:
            regex = r"<answer>\s*(\d+)\s*<\/answer>"
            match = re.search(regex, completion)
            solution = match.group(1).strip()
            answer = a
            if solution == answer:
                reward = 1.0
            else:
                reward = 0.0
        except Exception:
            pass
        print(f"Equation reward: {reward}")
        rewards.append(reward)
    return rewards