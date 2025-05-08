#!/usr/bin/env python3
import argparse
import pandas as pd
import time
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Query a model with optional adapter weights")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the model or model name to load"
    )
    parser.add_argument(
        "--adapter_repo",
        type=str,
        default=None,
        help="Hugging Face Hub repository ID for the adapter"
    )
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        default=None,
        help="JSONL file containing the prompts to send to the model"
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="Column name containing the prompts to send to the model"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=1.0,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k", 
        type=int, 
        default=50,
        help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="JSONL file to write the output to"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on (cuda or cpu)"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load the model in 8-bit precision"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the model in 4-bit precision"
    )
    return parser.parse_args()


def load_model(args):
    print(f"Loading model: {args.model}")
    start_time = time.time()
    
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=torch.float16 if not (args.load_in_8bit or args.load_in_4bit) else None,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # If an adapter is specified, load and merge it
    if args.adapter_repo:
        print(f"Loading adapter from {args.adapter_repo}")
        model = PeftModel.from_pretrained(
            model,
            args.adapter_repo,
            is_trainable=False
        )
        print("Adapter loaded successfully")
    
    model.eval()
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    return model, tokenizer


def query_model(model, tokenizer, prompt, args):
    print("Querying model...")
    start_time = time.time()
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    return output_text, inference_time


def main():
    args = parse_arguments()
    
    model, tokenizer = load_model(args)

    if not args.prompt_file:
        raise ValueError("Prompt file is required")

    df = pd.read_json(args.prompt_file, orient="records", lines=True)
    outputs = []
    total_inference_time = 0
    for i, row in df.iterrows():
        prompt = row[args.prompt_column]
        output, inference_time = query_model(model, tokenizer, prompt, args)
        total_inference_time += inference_time
        print("\n--- Generated Output ---")
        print(output)
        outputs.append({"prompt": prompt, "output": output, "inference_time": inference_time})
    
    if args.output_file:
        df = pd.DataFrame(outputs)
        df.to_json(args.output_file, orient="records", lines=True)
        print(f"Output saved to {args.output_file}")

    print(f"Total inference time: {total_inference_time:.2f} seconds")


if __name__ == "__main__":
    main()
