from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dataset = load_dataset("trl-lib/tldr", split="train")
eval_dataset = load_dataset("trl-lib/tldr", split="test")
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

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

# Add quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load the base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    model,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
).to(device)

# Apply LoRA to the model
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # Print the percentage of trainable parameters

training_args = GRPOConfig(
    output_dir="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    logging_steps=10,
    save_strategy="steps",        # Save by steps instead of epochs
    eval_strategy="steps",
    save_steps=500,              # Save checkpoint every 500 steps
    save_total_limit=3,          # Keep only the last 3 checkpoints
    load_best_model_at_end=True, # Load the best model when training ends
    metric_for_best_model="reward", # Use reward as the metric to track
    # Add memory optimization settings
    gradient_accumulation_steps=4,    # Accumulate gradients over 4 steps
    per_device_train_batch_size=1,    # Reduce batch size
    gradient_checkpointing=True,      # Enable gradient checkpointing
    max_grad_norm=0.3,               # Clip gradients to prevent memory spikes
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_len,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)
trainer.train()