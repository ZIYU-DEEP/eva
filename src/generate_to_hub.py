"""
Push the generated responses to the hub.
"""

from datasets import load_dataset, Dataset
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
    parser.add_argument("--n_pairs", type=int, default=5)
    parser.add_argument("--to_hf_dataset", type=str, 
                        default="cat-searcher/sppo-gemma-1.1-2b-it-split-0-all")
    
    return parser.parse_args()


def main():

    # -------------- Set up the arguments --------------- #
    args = parse_arguments()
    
    model_path = args.model_path
    dataset_name = args.dataset_name
    to_hf_dataset = args.to_hf_dataset
    
    n_pairs = args.n_pairs
    
    output_dir = Path(args.output_dir)  # Shared across files
    data_root = Path(args.data_root)
    gen_dir = data_root / 'generated' / output_dir


    # -------------- Set up the data --------------- #
    # Load the dataset
    data = load_dataset(dataset_name, split="train")

    # Set the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Get all the prompts
    prompts_all_raw = [data[idx]["prompt"] for idx in range(len(data))]
    print(prompts_all_raw[0])

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


    # -------------- Push to hub --------------- #
    # Create a DataFrame
    df = pd.DataFrame({'prompt': prompts_all_raw})
    for i in range(n_pairs):
        df[f'generate_{i}'] = [gen[i] for gen in candidates_texts]
    
    # Push to hub
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset.push_to_hub(to_hf_dataset, split='train', private=True)


if __name__ == "__main__":
    main()
