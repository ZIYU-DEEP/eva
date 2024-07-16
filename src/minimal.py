import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from typing import List, Tuple
import os
import pandas as pd
import tqdm

def setup_distributed(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    dist.destroy_process_group()

def hf_reward(
    prompt: str, 
    responses: List[str],  # List of responses for the prompt
    reward_model_path: str,
    max_tokens_hf: int = 2048,
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

def process_dataset(rank: int, world_size: int, dataset, reward_model_path: str):
    setup_distributed(rank, world_size)

    # Split dataset
    n_samples = len(dataset)
    per_gpu_samples = (n_samples + world_size - 1) // world_size  # Ensure at least one sample per GPU
    start_idx = rank * per_gpu_samples
    end_idx = min(start_idx + per_gpu_samples, n_samples)

    df = dataset.select(range(start_idx, end_idx)).to_pandas()

    all_rewards = []
    all_critiques = []

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc=f"Processing on GPU {rank}"):
        prompt = row['prompt']
        responses = [
            row[f'generate_{i}'] for i in range(5)  # TODO: better make this a param
        ]
        
        rewards = hf_reward(prompt, [r[1]['content'] for r in responses], reward_model_path)
        row_rewards, row_critiques = zip(*rewards)
        
        all_rewards.append(row_rewards)
        all_critiques.append(row_critiques)

    cleanup_distributed()

    # Save results for this GPU
    results_df = pd.DataFrame({
        'rewards': all_rewards,
        'critiques': all_critiques
    })
    results_df.to_csv(f'results_gpu_{rank}.csv', index=False)


def main():
    world_size = torch.cuda.device_count()
    # Load the dataset
    dataset = load_dataset("cat-searcher/responses-gemma-1.1-2b-it-split-0-all")
    # Slice the first 100 rows as a subset
    subset = dataset['train'].select(range(100))
    
    reward_model_path = "RLHFlow/ArmoRM-Llama3-8B-v0.1" 

    mp.spawn(
        process_dataset,
        args=(world_size, subset, reward_model_path),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()

