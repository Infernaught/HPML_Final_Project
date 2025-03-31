from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

dataset = load_dataset("trl-lib/tldr", split="train")
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

training_args = GRPOConfig(output_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", logging_steps=10)
trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()