"""
Generate Y | X with a fixed model checkpoint.

Input: the dataset and models on huggingface.
Output: json files with the generated responses.
        named as responses_{i}_{j}.json,
        where i is the GPU index and j is the response index.
"""

from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm

import argparse
import torch
import json
import os
from pathlib import Path
import random
import warnings
import numpy as np
import math

warnings.filterwarnings("ignore")
    

def parse_arguments():
    """
    Parse command line arguments.
    """
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, 
                        default="google/gemma-2-9b-it")
    parser.add_argument("--dataset_name", type=str, 
                        default="cat-searcher/ultrafeedback-split-1")
    parser.add_argument("--output_dir", type=str, 
                        default="responses-gemma-1.1-2b-it-split-1-buffer",
                        help="Directory to save the responses.")
    parser.add_argument("--data_root", type=str, 
                        default="./data")
    parser.add_argument("--n_gpus", type=int, default=1)
    parser.add_argument("--vllm_world_size", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--n_pairs", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    return parser.parse_args()


def set_seed(seed=5775709):
    """
    Set up the random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def apply_template(text, tokenizer):
    """
    Apply chat template to the tokenizer.
    """
    
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}, 
         {"role": "assistant", "content": "None"}],
        tokenize=False, 
        add_generate_prompt=True,
    ).split("None")[0]


def split_prompts(prompts, n_gpus, local_rank):
    """
    Split the prompts into chunks for distributed processing across multiple GPUs.

    Args:
        prompts (list): The full list of prompts to be split.
        n_gpus (int): The total number of GPUs to split the work across.
        local_rank (int): The index of the fraction to return (0 to n_gpus-1).

    Returns:
        list: A subset of the prompts for the specified GPU to process.
    """
    n_prompts = len(prompts)
    frac_len = math.ceil(n_prompts / n_gpus)
    start = local_rank * frac_len
    end = min((local_rank + 1) * frac_len, n_prompts)
    return prompts[start:end]


def main():
    
    # -------------- Set up the arguments --------------- #
    args = parse_arguments()
    
    model_path = args.model_path
    dataset_name = args.dataset_name
    
    n_gpus = args.n_gpus
    vllm_world_size = args.vllm_world_size
    local_rank = args.local_rank
    
    n_pairs = args.n_pairs
    max_tokens = args.max_tokens
    dtype = args.dtype
    
    temperature = args.temperature
    top_p = args.top_p
    
    output_dir = Path(args.output_dir)  # Shared across files
    data_root = Path(args.data_root)
    gen_dir = data_root / 'generated' / output_dir
    gen_dir.mkdir(parents=True, exist_ok=True)


    # -------------- Set up the data --------------- #
    # Load the dataset
    data = load_dataset(dataset_name, split="train")

    # Set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Apply template to the prompt
    print(dataset_name)
    print(model_path)
    prompts_raw = [data[idx]["prompt"] for idx in range(len(data))]
    breakpoint()
    prompts = [apply_template(prompt, tokenizer) for prompt in prompts_raw]
    
    # prompts = [apply_template(data[idx]["prompt"], tokenizer) 
    #            for idx in range(len(data))]
    
    # Check the prompts
    print(prompts[0])
    
    # Split the prompts to fit in multiple gpus
    prompts = split_prompts(prompts=prompts, 
                            n_gpus=n_gpus,
                            local_rank=local_rank)


    # -------------- Set up the model --------------- #
    # Set the language model
    llm = LLM(
        model=model_path,
        tensor_parallel_size=vllm_world_size,
        dtype=dtype,
    )
    
    # Generate responses
    for p in range(n_pairs):
        set_seed(p * 50)
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,  # Maximum number of tokens to generate
            seed=p * 50,
        )
        
        # Batch generation
        response = llm.generate(prompts, sampling_params)
        
        # Extract the text part
        output = list(map(lambda x: x.outputs[0].text, response))
        
        # Save the responses
        with open(gen_dir / f'responses_{local_rank}_{p}.json', 'w') as f:
            json.dump(output, f)


if __name__ == "__main__":
    main()
