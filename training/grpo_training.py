import sys
import os
import datetime
# Add the root directory to sys.path so Python can find 'tasks'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import argparse
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from torch.profiler import profile, record_function, ProfilerActivity
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
from training.constants import AVAILABLE_MODELS
from training.reward_functions import reward_function_mapping
import wandb

# argparse for selecting base model and quantization
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="phi", choices=AVAILABLE_MODELS.keys())

parser.add_argument("--quantize", action="store_true", help="Enable 4-bit quantization")
parser.add_argument("--quant_type", default="nf4", choices=["nf4", "fp4"], help="4-bit quantization type")
parser.add_argument("--double_quant", action="store_true", help="Use double quantization (nested)")
parser.add_argument("--compute_dtype", default="float16", choices=["float16", "bfloat16"], help="Compute dtype")
parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization instead of 4-bit")
parser.add_argument("--task", choices=["aime", "countdown"], help="Task to train on")
parser.add_argument("--train_dataset_path", help="Path to train dataset")
parser.add_argument("--eval_dataset_path", help="Path to eval dataset")

args = parser.parse_args()
BASE_MODEL = AVAILABLE_MODELS[args.model]

# Initialize wandb
wandb.init(
    project="grpo-training",  # Name of your project
    name=f"grpo-{BASE_MODEL}-{args.task}",  # Name of this specific run
    config={
        "model": BASE_MODEL,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "quantize": args.quantize,
        "quant_type": args.quant_type,
        "double_quant": args.double_quant,
        "compute_dtype": args.compute_dtype,
        "use_8bit": args.use_8bit,
    }
)

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
        
    def on_train_begin(self, args, state, control, **kwargs):
        # Optionally start the profiler here
        if hasattr(self.profiler, "start"):
            self.profiler.start()
    
    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()
        
    def on_train_end(self, args, state, control, **kwargs):
        # Make sure to stop the profiler
        if hasattr(self.profiler, "stop"):
            self.profiler.stop()

# torch.cuda.memory._record_memory_history() # will record memory allocation over time (then save to pickle at end)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
dataset = pd.read_json(args.train_dataset_path, orient="records", lines=True)
eval_dataset = pd.read_json(args.eval_dataset_path, orient="records", lines=True)
dataset = Dataset.from_pandas(dataset)
eval_dataset = Dataset.from_pandas(eval_dataset)

reward_functions = reward_function_mapping[args.task]

# Define LoRA configuration
lora_config = LoraConfig(
    r=16,                     # Rank
    lora_alpha=32,           # Alpha scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which modules to apply LoRA to
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Add quantization config with argparse
if args.quantize:
    from transformers import BitsAndBytesConfig

    compute_dtype = getattr(torch, args.compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=not args.use_8bit,
        load_in_8bit=args.use_8bit,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.double_quant,
    )

    print("Quantization Enabled:", bnb_config)
else:
    bnb_config = None
    print("Not using quantization")

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
    output_dir=f"outputs/{BASE_MODEL}/{args.task}",
    logging_steps=5,
    save_strategy="steps",        # Save by steps instead of epochs
    eval_strategy="steps",
    save_steps=5,              # Save checkpoint every 50 steps
    save_total_limit=3,          # Keep only the last 3 checkpoints
    load_best_model_at_end=True, # Load the best model when training ends
    metric_for_best_model="loss", # Use reward as the metric to track
    max_completion_length=256,
    # Add memory optimization settings
    gradient_accumulation_steps=4,    # Accumulate gradients over 4 steps
    per_device_train_batch_size=4,    
    per_device_eval_batch_size=4,
    gradient_checkpointing=True,      # Enable gradient checkpointing
    max_grad_norm=0.3,               # Clip gradients to prevent memory spikes
    num_generations=4,
    # Add wandb reporting
    report_to="wandb",
    run_name=wandb.run.name,
)

log_dir = "logs/profiler/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

profiler = profile(
    schedule=torch.profiler.schedule( # this schedule will capture 3 steps throughout training process
        wait=19,
        warmup=2,
        active=1,
        repeat=3,
    ),
    activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
    profile_memory=True, # only capture the memory as the rest are too expensive
    record_shapes=False,
    with_flops=False,
    with_stack=False,
    # on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir)
)


trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_functions,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    # callbacks=[ProfilerCallback(profiler)] # prints out memory allocation for each step
)

trainer.train()
# torch.cuda.memory._dump_snapshot("outputs/my_snapshot.pickle") # with profileing not needed

# Close wandb at the end
wandb.finish()