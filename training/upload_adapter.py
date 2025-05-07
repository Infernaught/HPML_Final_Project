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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Login to Hugging Face (requires HUGGING_FACE_TOKEN environment variable to be set)
    if "HUGGING_FACE_TOKEN" not in os.environ:
        print("Warning: HUGGING_FACE_TOKEN environment variable not set. Cannot upload to Hugging Face Hub.")
        return
    
    print("Logging in to Hugging Face Hub...")
    login(token=os.environ["HUGGING_FACE_TOKEN"])
    
    # Initialize the Hugging Face API client
    api = HfApi()
    
    # Upload the model to the Hub
    print(f"Creating repository: {args.repo_id}")
    api.create_repo(args.repo_id, private=args.private, exist_ok=True)
    
    # Push the model to the Hub
    print(f"Uploading model to: {args.repo_id}")
    api.upload_folder(
        folder_path=args.adapter_path,
        repo_id=args.repo_id,
        repo_type="model",
    )
    
    print(f"Model uploaded to: https://huggingface.co/{args.repo_id}")
    
if __name__ == "__main__":
    main()
