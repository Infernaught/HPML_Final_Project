#!/usr/bin/env python3
import argparse
import pandas as pd
import time
from typing import Dict, List, Optional, Union

from vllm import LLM, SamplingParams


def parse_arguments():
    parser = argparse.ArgumentParser(description="Serve and query a model using vllm")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to the model or model name to load"
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
        "--gpu_memory_utilization", 
        type=float, 
        default=0.9,
        help="GPU memory utilization target (0.0 to 1.0)"
    )
    parser.add_argument(
        "--tensor_parallel_size", 
        type=int, 
        default=1,
        help="Number of GPUs to use for tensor parallelism"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=None,
        help="JSONL file to write the output to"
    )
    parser.add_argument(
        "--quantization", 
        type=str, 
        choices=["awq", "gptq", "squeezellm", "None"],
        default="None",
        help="Quantization method to use (awq, gptq, squeezellm, or None)"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        choices=["float16", "bfloat16", "float32", "auto"],
        default="auto",
        help="Data type for model weights (float16, bfloat16, float32, or auto)"
    )
    parser.add_argument(
        "--cpu_offloading", 
        type=bool,
        default=False,
        help="Enable CPU offloading for some model layers"
    )
    return parser.parse_args()


def load_model(args):
    print(f"Loading model: {args.model}")
    start_time = time.time()
    
    # Convert quantization string to None if "None" is specified
    quantization = None if args.quantization == "None" else args.quantization
    
    model = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        quantization=quantization,
        dtype=args.dtype,
        cpu_offloading=args.cpu_offloading,
    )
    
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    return model


def get_prompt(args):
    if args.prompt:
        return args.prompt
    elif args.prompt_file:
        with open(args.prompt_file, "r") as f:
            return f.read()
    else:
        raise ValueError("Either --prompt or --prompt_file must be provided")


def query_model(model, prompt, args):
    print("Querying model...")
    start_time = time.time()
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )
    
    outputs = model.generate(prompt, sampling_params)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds")
    
    return outputs[0].outputs[0].text, inference_time


def main():
    args = parse_arguments()
    
    model = load_model(args)
    prompt = get_prompt(args)

    if not args.prompt_file:
        raise ValueError("Prompt file is required")

    df = pd.read_json(args.prompt_file, orient="records", lines=True)
    outputs = []
    total_inference_time = 0
    for i, row in df.iterrows():
        prompt = row[args.prompt_column]
        output, inference_time = query_model(model, prompt, args)
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
