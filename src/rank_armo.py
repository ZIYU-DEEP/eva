"""
Get the absolute rewards for each responses, and annotate the chosen and rejected responses to be used by contrastive preference learning.
"""

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset
from typing import List, Tuple
from pathlib import Path
import os
import pandas as pd
import numpy as np
import tqdm
import argparse


def setup_distributed(rank: int, world_size: int):
    """
    Prepare for DDP.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """
    Clean DDP.
    """
    dist.destroy_process_group()


def process_dataset(rank: int, 
                    world_size: int, 
                    dataset: dict, 
                    reward_model_path: str, 
                    torch_dtype: str,
                    n_generations: int=5,
                    batch_size: int=10):
    
    # Set up DDP
    setup_distributed(rank, world_size)

    # Split dataset
    n_samples = len(dataset)
    per_gpu_samples = (n_samples + world_size - 1) // world_size  # Ensure at least one sample per GPU
    start_idx = rank * per_gpu_samples
    end_idx = min(start_idx + per_gpu_samples, n_samples)

    # Create the dataframe per gpu
    df = dataset.select(range(start_idx, end_idx)).to_pandas()

    # Load progress if exists
    progress_file = f'./progress_gpu_{rank}.csv'
    if os.path.exists(progress_file):
        processed_indices = pd.read_csv(progress_file)['index'].tolist()
    else:
        processed_indices = []

    # Initialize the lists to store the results
    all_rewards = []
    all_critiques = []
    all_prompts = []
    all_responses = {f'generate_{i}': [] for i in range(n_generations)}

    # Initialize the model and tokenizer only once
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, 
        device_map='cuda', 
        trust_remote_code=True, 
        torch_dtype=torch_dtype).to(torch.device('cuda'))
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path, use_fast=True)

    # Populate the lists
    for idx, row in tqdm.tqdm(df.iterrows(), 
                              total=len(df), 
                              desc=f"Rewarding on GPU {rank}", 
                              disable=rank != 0):
        if idx in processed_indices:
            continue  # Skip already processed rows
        
        prompt = row['prompt']
        responses = [
            row[f'generate_{i}'] for i in range(n_generations) 
        ]
        
        # Calculate the rewards
        rewards = hf_reward(prompt=prompt, 
                            responses=[r[1]['content'] for r in responses], 
                            model=model, 
                            tokenizer=tokenizer)
        row_rewards, row_critiques = zip(*rewards)
        
        # Update the lists
        all_prompts.append(prompt)
        all_rewards.append(row_rewards)
        all_critiques.append(row_critiques)
        for i in range(n_generations): all_responses[f'generate_{i}'].append(responses[i])

        # Save progress
        processed_indices.append(idx)
        pd.DataFrame({'index': processed_indices}).to_csv(progress_file, index=False)
        
        # Save results periodically
        if len(processed_indices) % batch_size == 0:
            save_temp_results(rank, all_prompts, all_rewards, all_critiques, all_responses, n_generations)

    # Clean DDP
    cleanup_distributed()

    # Save final results
    save_temp_results(rank, all_prompts, all_rewards, all_critiques, all_responses, n_generations)

    # Remove progress file after completion
    if os.path.exists(progress_file):
        os.remove(progress_file)


def hf_reward(
    prompt: str, 
    responses: List[str], 
    model, 
    tokenizer) -> List[Tuple[float, str]]:
    
    """
    Calculate reward using a Hugging Face model.
    """
    if "armorm-llama3-8b" in model.config._name_or_path.lower():
        
        # Reformat the input
        messages = [
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response}]
            for response in responses
        ]
        
        # Apply the chat template for tokens
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", padding=True).to(torch.device('cuda'))
        
        # Get the scores
        with torch.no_grad():
            output = model(input_ids)
            scores = output.logits.cpu().float().tolist()
            
    else:
        raise NotImplementedError(
            f"Reward calculation not implemented for model: {model.config._name_or_path}")

    return list(zip(scores, [''] * len(scores)))


def leave_one_out(rewards):
    """
    Calculate the leave one out advantage for every reward.
    """
    n = len(rewards)
    leave_one_out_rewards = [
        rewards[i] - (sum(rewards) - rewards[i]) / (n - 1) for i in range(n)
    ]
    return np.mean(leave_one_out_rewards)


def save_temp_results(rank, 
                      all_prompts, 
                      all_rewards, 
                      all_critiques, 
                      all_responses,
                      n_generations):
    # -------------------------------------------------------------------
    # Get the dataframe
    results_df = pd.DataFrame({
        'prompt': all_prompts,
        'rewards': all_rewards,
        'critiques': all_critiques,
    })
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Add additional information
    results_df['reward_mean'] = results_df['rewards'].apply(np.mean)
    results_df['reward_var'] = results_df['rewards'].apply(np.var)
    results_df['reward_gap'] = results_df['rewards'].apply(lambda x: max(x) - min(x))
    results_df['reward_maxmean'] = results_df['rewards'].apply(
        lambda x: max(x) - np.mean(x))
    results_df['reward_loo'] = results_df['rewards'].apply(leave_one_out)

    # Add response columns
    for i in range(n_generations):
        results_df[f'generate_{i}'] = all_responses[f'generate_{i}']
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # Add chosen and rejected columns
    results_df['chosen'] = results_df.apply(
        lambda row: [
            {"role": "user", 
             "content": row['prompt']},
            {"role": "assistant", 
             "content": row[f'generate_{np.argmax(row["rewards"])}']}
        ],
        axis=1
    )
    results_df['rejected'] = results_df.apply(
        lambda row: [
            {"role": "user", 
             "content": row['prompt']},
            {"role": "assistant", 
             "content": row[f'generate_{np.argmin(row["rewards"])}']}
        ],
        axis=1
    )
    # -------------------------------------------------------------------

    # Save temporary results
    results_df.to_csv(f'./temp_results_gpu_{rank}.csv', index=False)
    results_df.to_parquet(f'./temp_results_gpu_{rank}.parquet', index=False)


def combine_results(world_size: int,
                    df_path: str,
                    parquet_path: str):
    """
    Combine results from different GPUs in DDP.
    """
    combined_df = pd.concat(
        [pd.read_csv(f'./temp_results_gpu_{rank}.csv') for rank in range(world_size)],
        ignore_index=True
    )
    combined_df.to_csv(df_path, index=False)
    combined_df.to_parquet(parquet_path, index=False)

    # Clean up temporary files
    for rank in range(world_size):
        os.remove(f'./temp_results_gpu_{rank}.csv')
        os.remove(f'./temp_results_gpu_{rank}.parquet')


def push_to_hf(hf_username: str = 'cat-searcher',
               hf_reward_repo_name: str = 'responses-gemma-1.1-2b-it-split-0-rewards',
               parquet_path: str = './data/rewards/output_dir/reward.parquet'):
    """
    Push to huggingface repo.
    """
    repo = Dataset.from_parquet(str(parquet_path))
    repo.push_to_hub(f'{hf_username}/{hf_reward_repo_name}', 
                     split='train',
                     private=True)
    

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, 
                        default="cat-searcher/ultrafeedback-gemma-2-9b-it-split-1-all",
                        help='The dataset with prompts and multiple response generations.')
    parser.add_argument("--n_generations", type=int, default=6,
                        help='The number of response generations in the dataset.')
    parser.add_argument("--output_dir", type=str, 
                        default="ultrafeedback-gemma-2-9b-it-split-1")
    parser.add_argument("--data_root", type=str, 
                        default="./data")
    parser.add_argument("--hf_username", type=str, 
                        default="cat-searcher",
                        help='The username to push the results to on Hugging Face.')
    parser.add_argument("--reward_model_path", type=str, 
                        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
                        help='The reward function used to judge the response.')
    parser.add_argument("--torch_dtype", type=str, default='bfloat16')

    return parser.parse_args()

def main():
    
    # -------------- Set up the arguments --------------- #
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    
    data_root = Path(args.data_root)
    reward_dir = data_root / 'eval' / output_dir
    reward_dir.mkdir(parents=True, exist_ok=True)
    
    filename_suffix = f"{args.reward_model_path.split('/')[-1]}"
    df_path = reward_dir / f'rewards_{filename_suffix}.csv'
    parquet_path = reward_dir / f'rewards_{filename_suffix}.parquet'
    hf_reward_repo_name = args.output_dir + "-all-hf-rewards"
    
    # -------------- Set up the datasets and get rewards --------------- #
    dataset = load_dataset(args.input_dataset)
    subset = dataset['train']
    # subset = dataset['train'].select(range(20))  # DEBUG TODO: Remove this line

    # Get the rewards
    world_size = torch.cuda.device_count()
    mp.spawn(
        process_dataset,
        args=(world_size, subset, args.reward_model_path, args.torch_dtype, args.n_generations),
        nprocs=world_size,
        join=True
    )

    # Combine the temporary files into a single CSV and Parquet file
    combine_results(world_size=world_size,
                    df_path=df_path,
                    parquet_path=parquet_path)

    # Push to Hugging Face
    push_to_hf(hf_username=args.hf_username,
               hf_reward_repo_name=hf_reward_repo_name,
               parquet_path=parquet_path)


if __name__ == "__main__":
    main()
