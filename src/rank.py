"""
Rank the responses for each prompt.

Input: prompts (list[str]) and responses (list[tuple]).
Output: a numpy array ranks,
        where ransks[i][j] represents the rank of 
        the j-th response for the i-th prompt.
"""

from datasets import load_dataset
import json
import pandas as pd
import argparse
import llm_blender
import os
import numpy as np
import warnings
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Tuple, Any
import math

warnings.filterwarnings("ignore")


def parse_arguments():
    """
    Parse command line arguments.
    """
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, 
                        default="google/gemma-1.1-2b-it")
    parser.add_argument("--dataset_name", type=str, 
                        default="cat-searcher/ultra-feedback-split-0")
    parser.add_argument("--output_dir", type=str, 
                        default="responses-gemma-1.1-2b-it-split-0")
    parser.add_argument("--data_root", type=str, 
                        default="./data")
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--n_pairs", type=int, default=5)
    
    return parser.parse_args()

def ranking(prompts: List[str], 
            candidates: List[Tuple[str, ...]],
            ranking_dir: str='./data/ranking/gemma',
            local_rank: int=0):
    """
    Rank the candidate models.
    """
    
    # Load the blender object
    blender = llm_blender.Blender()
    blender.loadranker("llm-blender/PairRM")
    
    # Rank the prompts by pair RM score
    ranks = blender.rank(prompts,     # note: the i-th split
                         candidates,  # note: the i-th split
                         return_scores=True, 
                         batch_size=1)
    
    # Save file (note each gpu takes care of a split of the data)
    filepath = Path(ranking_dir) / f'{local_rank}_{local_rank}.npy'
    np.save(filepath, ranks)


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


def main():

    # -------------- Set up the arguments --------------- #
    args = parse_arguments()
    
    model_path = args.model_path
    dataset_name = args.dataset_name
    
    n_gpus = args.n_gpus
    local_rank = args.local_rank
    n_pairs = args.n_pairs
    
    output_dir = Path(args.output_dir)  # Shared across files
    data_root = Path(args.data_root)
    gen_dir = data_root / 'generated' / output_dir
    ranking_dir = data_root / 'ranking' / output_dir
    ranking_dir.mkdir(parents=True, exist_ok=True)


    # -------------- Set up the data --------------- #
    # Load the dataset
    data = load_dataset(dataset_name, split="train")

    # Set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Get all the prompts
    prompts_all = [apply_template(data[idx]["prompt"], tokenizer) 
                   for idx in range(len(data))]
    print(prompts_all[0])

    # Get all the responses
    all_generated = []

    for i in range(n_pairs):
        file_path = gen_dir / f'responses_{i}.json'
        
        with open(file_path) as f:
            generated = json.load(f)
            all_generated.append(generated)

    candidates_texts = list(zip(*all_generated))  # Make it a list of tuples
    assert len(data) == len(candidates_texts)
    print(f'Length of data: {len(data)}')


    # -------------- Get the ranking --------------- #
    # Split prompts into chunks by gpu
    prompts_i = split_prompts(
        prompts=prompts_all, 
        n_gpus=n_gpus, 
        local_rank=local_rank)
    
    # Split responses into chunks by gpu
    candidates_texts_i = split_prompts(
        prompts=candidates_texts,
        n_gpus=n_gpus, 
        local_rank=local_rank)

    # Get the ranking and save the results
    ranking(prompts=prompts_i,
            candidates=candidates_texts_i,
            ranking_dir=ranking_dir,
            local_rank=local_rank)


if __name__ == "__main__":
    main()
