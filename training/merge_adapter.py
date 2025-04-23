import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM
from peft import PeftModel
from training.constants import BASE_MODEL
from huggingface_hub import HfApi, login
import os

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    trust_remote_code=True
)

# Load the adapter
adapter_model = PeftModel.from_pretrained(
    base_model,
    "./sft_outputs/microsoft/Phi-3.5-mini-instruct/checkpoint-570",
    is_trainable=False
)

# Merge adapter weights with base model
merged_model = adapter_model.merge_and_unload()

# Save the merged model
output_dir = "./merged_model"
merged_model.save_pretrained(output_dir)
print(f"Merged model saved to {output_dir}")

# Login to Hugging Face (requires HUGGING_FACE_TOKEN environment variable to be set)
login(token=os.environ["HUGGING_FACE_TOKEN"])

# Initialize the Hugging Face API client
api = HfApi()

# Upload the model to the Hub
repo_id = "Infernaught/phi-3.5-mini-aime"  # Replace with your desired repo name
api.create_repo(repo_id, private=False, exist_ok=True)

# Push the model to the Hub
api.upload_folder(
    folder_path=output_dir,
    repo_id=repo_id,
    repo_type="model",
)

print(f"Model uploaded to: https://huggingface.co/{repo_id}")
