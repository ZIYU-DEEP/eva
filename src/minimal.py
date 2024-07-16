import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset, Dataset
from typing import List, Tuple
import os
import pandas as pd
import tqdm
import argparse
from huggingface_hub import HfApi, Repository


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


def hf_reward(
    prompt: str, 
    responses: List[str],  # List of responses for the prompt
    reward_model_path: str,
    torch_dtype: str = 'bfloat16') -> List[Tuple[float, str]]:
    
    """
    Calculate reward using a Hugging Face model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, 
        device_map='cuda', 
        trust_remote_code=True, 
        torch_dtype=torch_dtype).to(torch.device('cuda'))

    tokenizer = AutoTokenizer.from_pretrained(reward_model_path, use_fast=True)

    if "armorm-llama3-8b" in reward_model_path.lower():
        
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
            f"Reward calculation not implemented for model: {reward_model_path}")

    return list(zip(scores, [''] * len(scores)))


def process_dataset(rank: int, 
                    world_size: int, 
                    dataset: dict, 
                    reward_model_path: str, 
                    torch_dtype: str,
                    n_generations: int=5):
    
    # Set up DDP
    setup_distributed(rank, world_size)

    # Split dataset
    n_samples = len(dataset)
    per_gpu_samples = (n_samples + world_size - 1) // world_size  # Ensure at least one sample per GPU
    start_idx = rank * per_gpu_samples
    end_idx = min(start_idx + per_gpu_samples, n_samples)

    # Create the dataframe per gpu
    df = dataset.select(range(start_idx, end_idx)).to_pandas()

    # Initialize the lists to store the results
    all_rewards = []
    all_critiques = []
    all_prompts = []

    # Populate the lists
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Processing on GPU {rank}"):
        prompt = row['prompt']
        responses = [
            row[f'generate_{i}'] for i in range(n_generations) 
        ]
        
        # Calculate the rewards
        rewards = hf_reward(prompt=prompt, 
                            responses=[r[1]['content'] for r in responses], 
                            reward_model_path=reward_model_path, 
                            torch_dtype=torch_dtype)
        row_rewards, row_critiques = zip(*rewards)
        
        # Update the lists
        all_prompts.append(prompt)
        all_rewards.append(row_rewards)
        all_critiques.append(row_critiques)

    # Clean DDP
    cleanup_distributed()

    # Save results for this GPU in a temporary file
    results_df = pd.DataFrame({
        'prompt': all_prompts,
        'rewards': all_rewards,
        'critiques': all_critiques
    })
    results_df.to_csv(f'./local/temp_results_gpu_{rank}.csv', index=False)
    results_df.to_parquet(f'./local/temp_results_gpu_{rank}.parquet', index=False)


def combine_results(world_size: int):
    """
    Combine results from different GPUs in DDP.
    """
    combined_df = pd.concat(
        [pd.read_csv(f'./local/temp_results_gpu_{rank}.csv') for rank in range(world_size)],
        ignore_index=True
    )
    combined_df.to_csv('./local/results_all_gpus.csv', index=False)
    combined_df.to_parquet('./local/results_all_gpus.parquet', index=False)

    # Clean up temporary files
    for rank in range(world_size):
        os.remove(f'./local/temp_results_gpu_{rank}.csv')
        os.remove(f'./local/temp_results_gpu_{rank}.parquet')


def push_to_hf(repo_name: str, hf_username: str):
    """
    Push to huggingface repo.
    """
    api = HfApi()
    repo_url = api.create_repo(name=repo_name, organization=hf_username, private=False, exist_ok=True)
    repo = Repository(local_dir="./local", clone_from=repo_url)

    repo.git_add(pattern="results_all_gpus.csv")
    repo.git_add(pattern="results_all_gpus.parquet")
    repo.git_commit("Add combined results from all GPUs")
    repo.git_push()

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, 
                        default="cat-searcher/responses-gemma-1.1-2b-it-split-0-all",
                        help='The dataset with prompts and multiple response generations.')
    parser.add_argument("--n_generations", type=int, default=5,
                        help='The number of response generations in the datast.')
    parser.add_argument("--hf_username", type=str, 
                        default="cat-searcher",
                        help='The username to push the results to on Hugging Face.')
    parser.add_argument("--reward_model_path", type=str, 
                        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
                        help='The reward function used to judge the response.')
    parser.add_argument("--torch_dtype", type=str, default='bfloat16')

    return parser.parse_args()

def main():
    
    args = parse_arguments()
    world_size = torch.cuda.device_count()
    dataset = load_dataset(args.input_dataset)
    subset = dataset['train'].select(range(100))  # DEBUG TODO: Remove this line

    # Get the rewards
    mp.spawn(
        process_dataset,
        args=(world_size, subset, args.reward_model_path, args.torch_dtype, args.n_generations),
        nprocs=world_size,
        join=True
    )

    # Combine the temporary files into a single CSV and Parquet file
    combine_results(world_size)

    # Push to Hugging Face
    push_to_hf(args.input_dataset, args.hf_username)


if __name__ == "__main__":
    main()
