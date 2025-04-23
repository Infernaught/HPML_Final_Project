import sys
import os
# Add the root directory to sys.path so Python can find 'tasks'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from torch.profiler import profile, record_function, ProfilerActivity
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
from training.constants import BASE_MODEL
from training.reward_functions import reward_function_mapping


# create a function to print out memory allocations
def gpu_memory_report(print_stmt = ""):
    # get allocations (current and max seen) (bytes) and convert to Gigabytes
    allocated_Gbytes = torch.cuda.memory_allocated() * 1e-9
    max_allocated_Gbytes = torch.cuda.max_memory_allocated() * 1e-9
    print(f"{print_stmt} GPU Memory Allocated: {allocated_Gbytes:.5f} GBs")
    print(f"{print_stmt} GPU Memory Max Allocated: {max_allocated_Gbytes:.5f} GBs")

# Want to be able to call this inside the training loop. Can use Huggingface Callback
# class GPUMemoryCallback(TrainerCallback):
#     def on_step_end(self, args, state, control, **kwargs):
#         gpu_memory_report(f"Step {state.global_step} - ")

class ProfilerCallback(TrainerCallback):
    def __init__(self, profiler):
        self.profiler = profiler
    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()

# torch.cuda.memory._record_memory_history() # will record memory allocation over time (then save to pickle at end)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dataset = pd.read_json("../tasks/aime/aime_train_dataset.jsonl", orient="records", lines=True)
eval_dataset = pd.read_json("../tasks/aime/aime_eval_dataset.jsonl", orient="records", lines=True)
dataset = Dataset.from_pandas(dataset)
eval_dataset = Dataset.from_pandas(eval_dataset)

reward_functions = reward_function_mapping["aime"]

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

# torch.cuda.reset_peak_memory_stats() # reset max GPU Memory allocation (profile = not needed)

# Load the base model with quantization
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager", # - use flash_attention_2 for A100/H100
).to(device)
gpu_memory_report("After Loading Model -")
# Apply LoRA to the model
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()  # Print the percentage of trainable parameters

gpu_memory_report("After LoRA -")

training_args = GRPOConfig(
    output_dir=f"outputs/{BASE_MODEL}_2",
    logging_steps=5,
    save_strategy="steps",        # Save by steps instead of epochs
    eval_strategy="steps",
    save_steps=5,              # Save checkpoint every 50 steps
    save_total_limit=3,          # Keep only the last 3 checkpoints
    load_best_model_at_end=True, # Load the best model when training ends
    metric_for_best_model="loss", # Use reward as the metric to track
    # Add memory optimization settings
    gradient_accumulation_steps=4,    # Accumulate gradients over 4 steps
    per_device_train_batch_size=16,    # Use a batch size of 16
    per_device_eval_batch_size=16,     # Use a batch size of 16 for evaluation
    gradient_checkpointing=True,      # Enable gradient checkpointing
    max_grad_norm=0.3,               # Clip gradients to prevent memory spikes
    num_generations=16,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_functions,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    callbacks=[ProfilerCallback(profile)] # prints out memory allocation for each step
)
with profile(
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=1,
        repeat=1,
    ),
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("logs/profiler")  # <- Saves trace files
) as prof:
    with record_function("grpo_training"):
        trainer.train()
# torch.cuda.memory._dump_snapshot("outputs/my_snapshot.pickle") # with profileing not needed