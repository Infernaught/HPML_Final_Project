from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dataset = load_dataset("trl-lib/tldr", split="train")
eval_dataset = load_dataset("trl-lib/tldr", split="test")
model = "Qwen/Qwen2.5-1.5B-Instruct"

# Define the reward function, which rewards completions that are close to 20 characters
def reward_len(completions, **kwargs):
    return [-abs(20 - len(completion)) for completion in completions]

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,                     # Rank
    lora_alpha=32,           # Alpha scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which modules to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model).to(device)

# Apply LoRA to the model
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # Print the percentage of trainable parameters

training_args = GRPOConfig(
    output_dir="Qwen/Qwen2.5-1.5B-Instruct",
    logging_steps=10,
    save_strategy="steps",        # Save by steps instead of epochs
    eval_strategy="steps",
    save_steps=500,              # Save checkpoint every 500 steps
    save_total_limit=3,          # Keep only the last 3 checkpoints
    load_best_model_at_end=True, # Load the best model when training ends
    metric_for_best_model="reward", # Use reward as the metric to track
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)
trainer.train()