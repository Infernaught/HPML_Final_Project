import sys
import os
# Add the root directory to sys.path so Python can find 'tasks'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
from training.constants import BASE_MODEL, AIME_TRAIN_DATASET_PATH_DISTILLED

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dataset = pd.read_json(AIME_TRAIN_DATASET_PATH_DISTILLED, orient="records", lines=True)
dataset = Dataset.from_pandas(dataset)

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,                    
    lora_alpha=32,           
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
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
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager", # - use flash_attention_2 for A100/H100
).to(device)

# Apply LoRA to the model
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # Print the percentage of trainable parameters

training_args = SFTConfig(
    output_dir=f"sft_outputs/{BASE_MODEL}",
    logging_steps=5,
    save_strategy="steps",
    save_steps=5,
    save_total_limit=3,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)
trainer.train()