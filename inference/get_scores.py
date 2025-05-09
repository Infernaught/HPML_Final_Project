#!/usr/bin/env python3
import argparse
import pandas as pd
from typing import List, Dict
import sys
import os

# Add the root directory to sys.path so Python can find 'tasks'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.aime.reward_functions import equation_reward_func as aime_equation_reward
from tasks.countdown.reward_functions import equation_reward_func as countdown_equation_reward

def parse_arguments():
    parser = argparse.ArgumentParser(description="Score model outputs using task-specific reward functions")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="JSONL file containing model outputs to score"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["aime", "countdown"],
        help="Task to evaluate (aime or countdown)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="JSONL file to write scored outputs to"
    )
    return parser.parse_args()

def score_aime_outputs(df: pd.DataFrame) -> List[Dict]:
    """Score outputs for the AIME task using all reward functions."""
    prompts = df["prompt"].tolist()
    outputs = df["output"].tolist()
    
    # The answers should be in the same order as the prompts
    with open("tasks/aime/aime_eval_dataset.jsonl", "r") as f:
        eval_df = pd.read_json(f, orient="records", lines=True)
    eval_answers = eval_df["answer"].tolist()

    # Get scores from each reward function
    equation_scores = aime_equation_reward(prompts, outputs, eval_answers)
    
    # Combine scores into results
    results = []
    for i, row in df.iterrows():
        result = {
            "prompt": row["prompt"],
            "output": row["output"],
            "inference_time": row["inference_time"],
            "equation_score": equation_scores[i],
        }
        results.append(result)
    
    return results

def score_countdown_outputs(df: pd.DataFrame) -> List[Dict]:
    """Score outputs for the Countdown task using all reward functions."""
    prompts = df["prompt"].tolist()
    outputs = df["output"].tolist()
    
    # For Countdown, we need to extract numbers and target from the prompt
    # The format is typically: "Numbers: [n1, n2, ...] Target: target"
    nums = []
    targets = []
    for prompt in prompts:
        # Extract numbers and target using string manipulation
        numbers_list = prompt.split("Using the numbers [")[1].split("]")[0].strip().split(", ")
        target = prompt.split("create an equation that equals ")[1].split(".")[0].strip()
        nums.append(numbers_list)  # Convert string representation of list to actual list
        targets.append(int(target))
    
    # Get scores from each reward function
    equation_scores = countdown_equation_reward(prompts, outputs, nums, targets)
    
    # Combine scores into results
    results = []
    for i, row in df.iterrows():
        result = {
            "prompt": row["prompt"],
            "output": row["output"],
            "inference_time": row["inference_time"],
            "equation_score": equation_scores[i],
        }
        results.append(result)
    
    return results

def main():
    args = parse_arguments()
    
    # Read the input file
    df = pd.read_json(args.input_file, orient="records", lines=True)
    
    # Score outputs based on task
    if args.task == "aime":
        results = score_aime_outputs(df)
    else:  # countdown
        results = score_countdown_outputs(df)
    
    # Calculate and print summary statistics
    total_score = sum(r["equation_score"] for r in results)
    print(f"\nScoring Summary for {args.task.upper()} task:")
    print(f"Total score: {total_score}")
    print("\nDetailed Scores:")
    for i, result in enumerate(results):
        print(f"\nExample {i+1}:")
        print(f"Prompt: {result['prompt']}")
        print(f"Output: {result['output']}")
        print(f"Equation score: {result['equation_score']}")
    
    # Save results if output file is specified
    if args.output_file:
        output_df = pd.DataFrame(results)
        output_df.to_json(args.output_file, orient="records", lines=True)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()
