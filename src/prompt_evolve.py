"""
Evaluate prompts and their responses, calculate rewards, and sample based on metrics.
"""

import os
import re
import time
import openai
import json
import tqdm
import torch
import argparse
import numpy as np
import pandas as pd
import concurrent.futures

from pathlib import Path
from openai import OpenAI
from typing import List, Tuple, Dict
from datasets import load_dataset, Dataset
from typing import Callable, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForSequenceClassification, AutoTokenizer



def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, 
                        default="responses-gemma-1.1-2b-it-split-0-all")
    parser.add_argument("--hf_username", type=str, 
                        default="cat-searcher")
    parser.add_argument("--reward_function", type=str, default="openai")
    parser.add_argument("--sample_metric", type=str, default="reward_mean")
    parser.add_argument("--sample_frac", type=float, default=0.25)
    parser.add_argument("--reward_model_path", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--torch_dtype", type=str, default='bfloat16')
    parser.add_argument("--evaluation_mode", type=str, 
                        choices=['individual', 'batch'], 
                        default='batch',
    )
    return parser.parse_args()


def openai_reward_individual(
    prompt: str, 
    response: str,
    reward_model_path: str = 'gpt-4-0125-preview',
    max_tokens_openai: str = 8192,
    temperature_openai: float = 0.2) -> List[Tuple[float, str]]:
    """
    Calculate reward for a single response using OpenAI's GPT-4.
    """
    
    system_prompt = """You are an AI assistant tasked with evaluating the quality of responses to prompts. 
    Rate the response on a scale of 1 to 5, where 1 is the worst and 5 is the best. 
    Consider the following factors in your evaluation:
    a. Relevance: How well does the response address the prompt?
    b. Accuracy: Is the information provided correct and factual?
    c. Helpfulness: Does the response provide useful information or insights?
    d. Harmfulness: Does the response avoid causing harm, offense, or providing dangerous advice?
    e. Truthfulness: Is the information in the response truthful and not misleading? Is it honest?
    f. Conciseness: Is the response free of unnecessary information and to the point?
    
    You should provide ONE numeric score on a scale of 1 to 5 to access the OVERALL quality of the response. Your response should start with "#SCORE: "

    After providing the score for the response, give a concise and specific hint for future improvement for the given prompt. The hint should start with "#HINT: ". The hint should help language models to better respond to this prompt; the hint may specify the criteria to consider or some general rationale to follow. Additionally, the hint may mention any bad practices observed in the response that should be avoided.
    
    In summary, your response should be in the following format:
    #SCORE: {a number between 1 and 5}
    #HINT: {a few sentences for the language model to better respond to the prompt}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Prompt: {prompt}\n\nResponse: {response}\n\nPlease rate this response and provide a HINT:"}
    ]
    
    client = OpenAI()
    completion = client.chat.completions.create(
        model=reward_model_path,
        messages=messages,
        temperature=temperature_openai,
        max_tokens=max_tokens_openai,
    )
    content = completion.choices[0].message['content'].strip()
    
    # Extract score and hint
    score_match = re.search(r"#SCORE:\s*(\d+(?:\.\d+)?)", content)
    hint_match = re.search(r"#HINT:\s*(.*)", content, re.DOTALL)
    
    if score_match and hint_match:
        score = float(score_match.group(1))
        hint = hint_match.group(1).strip()
        return score / 5, hint  # Normalize to 0-1 range
    else:
        print(f"Error parsing OpenAI response: {content}")
        return 0.0, 'Error parsing critiques.'



def openai_reward_batch(
    prompt: str, 
    responses: List[str],
    reward_model_path: str = 'gpt-4-0125-preview',
    max_tokens_openai: str = 8192,
    temperature_openai: float = 0.2) -> List[Tuple[float, str]]:
    """
    Calculate rewards for multiple responses using OpenAI's GPT-4.
    """
    
    system_prompt = """You are an AI assistant tasked with evaluating the quality of multiple responses to a single prompt. 
    Rate each response on a scale of 1 to 5, where 1 is the worst and 5 is the best. 
    Consider the following major factors in your evaluation:
    a. Relevance: How well does the response address the prompt?
    b. Accuracy: Is the information provided correct and factual?
    c. Helpfulness: Does the response provide useful information or insights?
    d. Harmfulness: Does the response avoid causing harm, offense, or providing dangerous advice?
    e. Truthfulness: Is the information in the response truthful and not misleading? Is it honest?
    f. Conciseness: Is the response free of unnecessary information and to the point?
    
    Provide one score per response in the following format:
    #SCORE1: {score for response 1}
    #SCORE2: {score for response 2}
    #SCORE3: {score for response 3}
    ...

    After providing scores for the response, suggest ONE concise hint to help language models to better respond to this prompt. The hint should start with "#HINT: "; the hint may specify important criteria to consider or better rationales to follow. Additionally, the hint may mention any bad practices observed in the response that should be avoided. The hint should be concise and clear, DO NOT BE VERBOSE.

    In summary, your response should be in the following format:
    #SCORE1: {a number between 1 and 5}
    #SCORE2: {a number between 1 and 5}
    ... (add more scores here)
    #HINT: {a few sentences for the language model to better respond to the prompt}
    """
    
    messages = [
        {"role": "system", 
         "content": system_prompt},
        
        {"role": "user", 
         "content": f"Prompt: {prompt}\n\n" + "\n\n".join([f"Response {i+1}: {response}" for i, response in enumerate(responses)]) + "\n\nPlease rate these responses and provide a general HINT:"}
    ]
    
    client = OpenAI()
    completion = client.chat.completions.create(
        model=reward_model_path,
        messages=messages,
        temperature=temperature_openai,
        max_tokens=max_tokens_openai,
    )
    content = completion.choices[0].message['content'].strip()
    
    # Extract scores and hint
    scores = [float(score) / 5 
                for score in re.findall(r"#SCORE\d+:\s*(\d+(?:\.\d+)?)", 
                                        content)]
    hint_match = re.search(r"#HINT:\s*(.*)", content, re.DOTALL)
    hint = hint_match.group(1).strip() if hint_match else ''
    
    if len(scores) == len(responses):
        return list(zip(scores, [hint] * len(scores)))
    else:
        print(f"Error parsing OpenAI response: {content}")
        return [(0.0, "Error parsing critiques.")] * len(responses)  # To be filtered out


def hf_reward(
    prompt: str, 
    responses: List[str],  # List of responses for the prompt
    reward_model_path: str,
    max_tokens_hf: int = 2048,
    torch_dtype: str = 'bfloat16') -> Dict[str, float]:
    """
    Calculate reward using a Hugging Face model.
    TODO: load this with different ranks.
    """
    
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, 
        device_map='cuda', 
        trust_remote_code=True, 
        torch_dtype=torch_dtype) 
    
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
            messages, return_tensors="pt", padding=True).to('cuda')
        
        # Get the scores
        with torch.no_grad():
            output = model(input_ids)
            scores = output.score.cpu().float().tolist()
            
    else:
        raise NotImplementedError(
            f"Reward calculation not implemented for model: {reward_model_path}")

    return list(zip(scores, [''] * len(scores)))


def get_reward_function(reward_function: str = 'openai', 
                        evaluation_mode: str = 'batch',
                        reward_model_path: str = 'gpt-4-0125-preview',
                        max_tokens_openai: str = 8192,
                        temperature_openai: float = 0.2,
                        max_tokens_hf: int = 2048,
                        torch_dtype: str = 'bfloat16',
    ) -> Callable:
    """
    Return the specified reward function.
    """
    if reward_function == "openai":
        if evaluation_mode == 'batch':
            return lambda prompt, responses: openai_reward_batch(
                prompt, 
                responses, 
                reward_model_path,
                max_tokens_openai,
                temperature_openai,
            )
        else:
            return lambda prompt, response: openai_reward_individual(
                prompt, 
                response, 
                reward_model_path,
                max_tokens_openai,
                temperature_openai,
            )
    elif reward_function.startswith('hf'):
        return lambda prompt, response: hf_reward(
            prompt, 
            response, 
            reward_model_path,
            max_tokens_hf,
            torch_dtype)
    else:
        raise ValueError(f"Unknown reward function: {reward_function}")


def prompt_eval(
    dataset_name: str,
    hf_username: str,
    local_rank: int = 0,
    n_gpus: int = 8,
    reward_function: str = 'openai',
    evaluation_mode: str = 'batch',
    reward_model_path: str = 'gpt-4-0125-preview',
    max_tokens_openai: str = 8192,
    temperature_openai: float = 0.2,
    max_tokens_hf: int = 2048,
    torch_dtype: str = 'blfloat16',
) -> None:
    """
    Evaluate prompts and their responses, calculate rewards.
    """
    # Load dataset
    dataset = load_dataset(f"{hf_username}/{dataset_name}", split="train")
    
    # Get reward function
    reward_func = get_reward_function(reward_function, 
                                      evaluation_mode=evaluation_mode, 
                                      reward_model_path=reward_model_path,
                                      max_tokens_openai=max_tokens_openai,
                                      termperature_openai=temperature_openai,
                                      max_tokens_hf=max_tokens_hf,
                                      torch_dtype=torch_dtype)
    
    if reward_function.startswith("hf"):
        # Process only a portion of the dataset for Hugging Face models
        n_samples = len(dataset)
        start_idx = (n_samples // n_gpus) * local_rank
        end_idx = min((n_samples // n_gpus) * (local_rank + 1), n_samples)
        
        df = dataset.select(range(start_idx, end_idx)).to_pandas()
        
        all_rewards = []
        all_critiques = []
        
        for _, row in tqdm.tqdm(df.iterrows(), 
                                total=len(df), 
                                desc=f"Processing on GPU {local_rank}"):
            prompt = row['prompt']
            responses = [
                row[f'generate_{i}'] for i in range(5)  # TODO: better make this a param
            ]  
            
            rewards = reward_func(prompt, [r[1]['content'] for r in responses])
            row_rewards, row_critiques = zip(*rewards)
            
            all_rewards.append(row_rewards)
            all_critiques.append(row_critiques)
        
    else:  # OpenAI reward functions
        df = dataset.to_pandas()
        
        def process_row(row):
            prompt = row['prompt']
            responses = [row[f'generate_{i}'] for i in range(5)]
            
            if evaluation_mode == 'batch':
                batch_rewards = reward_func(prompt, [r[1]['content'] for r in responses])
                row_rewards, row_critiques = zip(*batch_rewards)
            else:
                row_rewards = []
                row_critiques = []
                for response in responses:
                    reward, critique = reward_func(prompt, response[1]['content'])
                    row_rewards.append(reward)
                    row_critiques.append(critique)
            
            return row_rewards, row_critiques

        max_workers = min(32, os.cpu_count() + 4) 
        all_rewards = []
        all_critiques = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_row, row) for _, row in df.iterrows()]
            
            for future in tqdm.tqdm(concurrent.futures.as_completed(futures), 
                                    total=len(futures), 
                                    desc="Processing with OpenAI"):
                rewards, critiques = future.result()
                all_rewards.append(rewards)
                all_critiques.append(critiques)

    # Add new columns
    df['rewards'] = all_rewards
    df['critiques'] = all_critiques
    df['reward_mean'] = df['rewards'].apply(np.mean)
    df['reward_var'] = df['rewards'].apply(np.var)
    df['reward_maxmin'] = df['rewards'].apply(lambda x: max(x) - min(x))

    # Save to parquet file
    output_file = f"{dataset_name}-re-{local_rank}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"Saved results to {output_file}")
    

def prompt_sample(
    dataset_name: str,
    hf_username: str = 'cat-searcher',
    metric: str = 'reward_mean',
    frac: float = 0.25,
) -> None:
    """
    Sample prompts based on a specified metric.
    """
    # Load dataset
    dataset = load_dataset(f"{hf_username}/{dataset_name}-re", split="train")
    df = dataset.to_pandas()

    # Calculate weights
    weights = df[metric] / df[metric].sum()

    # Sample rows
    sampled_df = df.sample(n=int(len(df) * frac), weights=weights, replace=False)

    # Save and push to hub
    new_dataset = Dataset.from_pandas(sampled_df)
    new_dataset.push_to_hub(
        f"{hf_username}/{dataset_name}-re-sample-{metric}", 
        split="train", 
        private=True)


def main():
    args = parse_arguments()

    # Evaluate prompts
    prompt_eval(
        dataset_name=args.input_dataset,
        hf_username=args.hf_username,
        reward_function=args.reward_function,
        nproc=args.nproc,
        batch_size=args.batch_size,
        evaluation_mode=args.evaluation_mode,
        reward_model_path=args.reward_model_path,
        max_tokens_openai=args.max_tokens_openai,
        temperature_openai=args.temperature_openai,
        max_tokens_hf=args.max_tokens_hf,
        torch_dtype=args.torch_dtype,
    )

    # Sample prompts
    prompt_sample(
        dataset_name=args.input_dataset,
        hf_username=args.hf_username,
        metric=args.sample_metric,
        frac=args.sample_frac,
    )


if __name__ == "__main__":
    main()