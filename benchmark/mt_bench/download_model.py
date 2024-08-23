"""
Just a faster way to have the model in cache.
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams

def load_model(model_path, num_gpus_total=8, dtype="bfloat16"):

    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus_total,
        dtype=dtype,
    )
    
    return llm

def main():
    parser = argparse.ArgumentParser(description='Load a transformer model.')
    parser.add_argument('--model-path', type=str, help='The name of the model to load')
    parser.add_argument('--num-gpus-total', type=int, default=8, help='The name of the model to load.')
    parser.add_argument('--dtype', type=str, default="bfloat16", help='The type of dtype.')
    args = parser.parse_args()

    load_model(args.model_path, args.num_gpus_total, args.dtype)
    print(f"Model {args.model_path} loaded successfully.")

if __name__ == "__main__":
    main()
