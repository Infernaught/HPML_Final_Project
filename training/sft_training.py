import sys
import os
# Add the root directory to sys.path so Python can find 'tasks'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from datasets import Dataset
from training.constants import AVAILABLE_MODELS, AIME_TRAIN_DATASET_PATH_DISTILLED
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=AVAILABLE_MODELS.keys(), required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--task", type=str, choices=["aime", "countdown"], required=True)
parser.add_argument("--train_dataset_path", type=str, required=True)
args = parser.parse_args()

BASE_MODEL = AVAILABLE_MODELS[args.model]

# Initialize wandb
wandb.init(
    project="sft-training",
    name=f"sft-{BASE_MODEL}-{args.task}",
    config={
        "model": BASE_MODEL,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
    }
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dataset = pd.read_json(args.train_dataset_path, orient="records", lines=True)
dataset = Dataset.from_pandas(dataset)

# Specific processing for countdown distilled dataset
countdown_prompts = []
if args.task == "countdown":
    for line in dataset:
        countdown_prompts.append(f"Using the numbers {line['nums']}, create an equation that equals {line['target']}. You can use basic arithmetic operations (+, -, *, /) and parentheses, and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.")
    dataset = dataset.add_column("prompt", countdown_prompts)

# Distilled datasets don't have the chat template applied yet, so we need to apply it
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
chat_templatted_dataset = []
for example in dataset:
    prompt = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
    chat_templatted_dataset.append({"prompt": prompt, "completion": example["completion"]})
dataset = Dataset.from_list(chat_templatted_dataset)
print(dataset[0])

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

output_dir = args.output_dir
if output_dir is None:
    output_dir = f"sft_outputs/{BASE_MODEL}/{args.task}"

training_args = SFTConfig(
    output_dir=output_dir,
    logging_steps=50,
    save_strategy="steps",
    save_steps=15,
    save_total_limit=3,
    per_device_train_batch_size=1,  
    gradient_accumulation_steps=4,  
    optim="paged_adamw_32bit",
    # Add wandb reporting
    report_to="wandb",
    run_name=wandb.run.name,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    schedule=torch.profiler.schedule(
        wait=1, warmup=1, active=3, repeat=1  # optional tuning
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./sft_logs/profiler")  # <- Saves trace files
) as prof:
    with record_function("sft_training"):
        trainer.train()

# Add this at the end of your script to properly close wandb
wandb.finish()
