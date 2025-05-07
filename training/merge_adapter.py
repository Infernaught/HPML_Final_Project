import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from training.constants import AVAILABLE_MODELS
from huggingface_hub import HfApi, login
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Merge adapter weights with base model and upload to Hugging Face Hub")
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        required=True,
        help="Path to the adapter model (e.g., './sft_outputs/microsoft/Phi-3.5-mini-instruct/checkpoint-570')"
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True,
        help="Hugging Face Hub repository ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--private", 
        action="store_true",
        help="Whether to make the repository private"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="model-merging",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        choices=AVAILABLE_MODELS.keys(),
        help="Base model name"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    BASE_MODEL = AVAILABLE_MODELS[args.base_model]
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=f"merge-{BASE_MODEL}-{os.path.basename(args.adapter_path)}",
        config={
            "base_model": BASE_MODEL,
            "adapter_path": args.adapter_path,
            "repo_id": args.repo_id,
            "private": args.private
        }
    )
    
    # Login to Hugging Face (requires HUGGING_FACE_TOKEN environment variable to be set)
    if "HUGGING_FACE_TOKEN" not in os.environ:
        print("Error: HUGGING_FACE_TOKEN environment variable not set. Cannot upload to Hugging Face Hub.")
        wandb.finish()
        return
    
    print("Logging in to Hugging Face Hub...")
    login(token=os.environ["HUGGING_FACE_TOKEN"])
    
    # Initialize the Hugging Face API client
    api = HfApi()
    
    # Create the repository
    print(f"Creating repository: {args.repo_id}")
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    
    print(f"Loading base model: {BASE_MODEL}")
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading adapter from: {args.adapter_path}")
    # Load the adapter
    adapter_model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        is_trainable=False
    )
    
    print("Merging adapter weights with base model...")
    # Merge adapter weights with base model
    merged_model = adapter_model.merge_and_unload()
    
    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Push the model and tokenizer directly to the Hub
    print(f"Uploading model to: {args.repo_id}")
    merged_model.push_to_hub(args.repo_id)
    tokenizer.push_to_hub(args.repo_id)
    
    print(f"Model uploaded to: https://huggingface.co/{args.repo_id}")
    
    # Log the model URL to wandb
    wandb.log({"model_url": f"https://huggingface.co/{args.repo_id}"})
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
