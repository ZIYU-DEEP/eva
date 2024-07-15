"""
Evaluate prompts and their responses, calculate rewards, and sample based on metrics.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from typing import Callable, List, Dict, Tuple
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re


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
    parser.add_argument("--hf_model_path", type=str, default="RLHFlow/ArmoRM-Llama3-8B-v0.1")
    parser.add_argument("--nproc", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--evaluation_mode", type=str, 
                        choices=['individual', 'comparative'], 
                        default='individual')
    return parser.parse_args()


def openai_reward_individual(
    prompt: str, 
    response: str,
    reward_model_path: str = 'gpt-4-0125-preview') -> List[Tuple[float, str]]:
    """
    Calculate reward for a single response using OpenAI's GPT-4.
    """
    
    system_prompt = """You are an AI assistant tasked with evaluating the quality of responses to prompts. 
    Rate the response on a scale of 1 to 5, where 1 is the worst and 5 is the best. 
    Consider the following factors in your evaluation:
    1. Relevance: How well does the response address the prompt?
    2. Accuracy: Is the information provided correct and factual?
    3. Coherence: Is the response well-structured and easy to understand?
    4. Helpfulness: Does the response provide useful information or insights?
    5. Completeness: Does the response fully answer all aspects of the prompt?
    6. Harmfulness: Does the response avoid causing harm, offense, or providing dangerous advice?
    7. Truthfulness: Is the information in the response truthful and not misleading?
    8. Conciseness: Is the response free of unnecessary information and to the point?
    9. Engagement: Is the response engaging and does it maintain the user interest?
    10. Tone: Is the tone of the response appropriate for the context?
    
    You should provide ONE numeric score on a scale of 1 to 5 to access the OVERALL quality of the response. Your response should start with "#SCORE: "

    After providing the score for the response, give a concise and specific hint for future improvement for the given prompt. The hint should start with "#HINT: ". The hint should help language models to better respond to this prompt; the hint may specify the criteria to consider or some general rationale to follow. Additionally, the hint may mention any bad practices observed in the response that should be avoided.
    
    In summary, your response should be in the following format:
    #SCORE: {a number between 1 and 5}
    #HINT: {a few sentences to improve the response}
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Prompt: {prompt}\n\nResponse: {response}\n\nPlease rate this response and provide a HINT:"}
    ]
    
    try:
        completion = openai.ChatCompletion.create(
            model=reward_model_path,
            messages=messages,
            temperature=0.2,
            max_tokens=300,
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
            return 0.0, "Error in parsing response"
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return 0.0, f"Error: {str(e)}"


def openai_reward_comparative(
    prompt: str, 
    responses: List[str],
    reward_model_path: str = 'gpt-4-0125-preview') -> List[Tuple[float, str]]:
    """
    Calculate rewards for multiple responses using OpenAI's GPT-4.
    """
    
    system_prompt = """You are an AI assistant tasked with evaluating the quality of multiple responses to a single prompt. 
    Rate each response on a scale of 1 to 5, where 1 is the worst and 5 is the best. 
    Consider the following factors in your evaluation:
    1. Relevance: How well does the response address the prompt?
    2. Accuracy: Is the information provided correct and factual?
    3. Coherence: Is the response well-structured and easy to understand?
    4. Helpfulness: Does the response provide useful information or insights?
    5. Completeness: Does the response fully answer all aspects of the prompt?
    6. Harmfulness: Does the response avoid causing harm, offense, or providing dangerous advice?
    7. Truthfulness: Is the information in the response truthful and not misleading?
    8. Conciseness: Is the response free of unnecessary information and to the point?
    9. Engagement: Is the response engaging and does it maintain the user’s interest?
    10. Tone: Is the tone of the response appropriate for the context?
    
    Provide scores for each response in the following format:
    #SCORE1: {score for response 1}
    #SCORE2: {score for response 2}
    ...

    After providing scores for the response, give ONE concise and specific hint for future improvement for the given prompt. The hint should start with "#HINT: ". The hint should help language models to better respond to this prompt; the hint may specify the criteria to consider or some general rationale to follow. Additionally, the hint may mention any bad practices observed in the response that should be avoided.

    In summary, your response should be in the following format:
    #SCORE1: {a number between 1 and 5}
    #SCORE2: {a number between 1 and 5}
    ... (add more scores here)
    #HINT: {a few sentences to improve the response}
    """
    
    messages = [
        {"role": "system", 
         "content": system_prompt},
        
        {"role": "user", 
         "content": f"Prompt: {prompt}\n\n" + "\n\n".join([f"Response {i+1}: {response}" for i, response in enumerate(responses)]) + "\n\nPlease rate these responses and provide a general HINT:"}
    ]
    
    try:
        completion = openai.ChatCompletion.create(
            model=reward_model_path,
            messages=messages,
            temperature=0.2,
            max_tokens=500,
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
            return [(0.0, "Error in parsing response")] * len(responses)
    except Exception as e:
        print(f"Error in OpenAI API call: {e}")
        return [(0.0, f"Error: {str(e)}")] * len(responses)


def huggingface_reward(
    prompt: str, 
    response: str, 
    reward_model_path: str) -> Dict[str, float]:
    """
    Calculate reward using a Hugging Face model.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, 
        device_map=device, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16)  # TODO: make it a paramter
    
    tokenizer = AutoTokenizer.from_pretrained(reward_model_path, use_fast=True)

    if "armorm-llama3-8b" in reward_model_path.lower():
        messages = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model(input_ids)
            multi_obj_rewards = output.rewards.cpu().float()
            gating_output = output.gating_output.cpu().float()
            preference_score = output.score.cpu().float()

        obj_transform = model.reward_transform_matrix.data.cpu().float()
        multi_obj_coeffs = gating_output @ obj_transform.T

        attributes = [
            'helpsteer-helpfulness', 'helpsteer-correctness', 'helpsteer-coherence',
            'helpsteer-complexity', 'helpsteer-verbosity', 'ultrafeedback-overall_score',
            'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
            'ultrafeedback-honesty', 'ultrafeedback-helpfulness', 'beavertails-is_safe',
            'prometheus-score', 'argilla-overall_quality', 'argilla-judge_lm', 'code-complexity',
            'code-style', 'code-explanation', 'code-instruction-following', 'code-readability']

        rewards_dict = {attr: score.item() 
                        for attr, score in zip(attributes, multi_obj_rewards[0])}
        rewards_dict['preference_score'] = preference_score.item()
    else:
        # Default behavior for other models
        inputs = tokenizer(
            prompt, 
            response, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,  # TODO: make it a parameter
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        rewards_dict = {"score": outputs.logits[0][1].item()}  # Assuming binary classification

    return rewards_dict


def get_reward_function(reward_function: str, **kwargs) -> Callable:
    """
    Return the specified reward function.
    """
    if reward_function == "openai":
        if kwargs.get('evaluation_mode') == 'comparative':
            return lambda prompt, responses: openai_reward_comparative(
                prompt, responses, kwargs.get('reward_model_path', 'gpt-4-0125-preview'))
        else:
            return lambda prompt, response: openai_reward_individual(
                prompt, response, kwargs.get('reward_model_path', 'gpt-4-0125-preview'))
    elif reward_function.startswith("huggingface/"):
        model_path = kwargs['model_path']
        return lambda prompt, response: huggingface_reward(prompt, response, model_path)
    else:
        raise ValueError(f"Unknown reward function: {reward_function}")


def process_batch(batch, reward_func, evaluation_mode):
    """
    Process a batch of prompts and responses.
    """
    rewards = []
    critiques = []
    for prompt, responses in zip(batch['prompt'], batch['responses']):
        if evaluation_mode == 'comparative':
            batch_rewards = reward_func(prompt, [r[1]['content'] for r in responses])
            row_rewards, row_critiques = zip(*batch_rewards)
        else:
            row_rewards = []
            row_critiques = []
            for response in responses:
                reward, critique = reward_func(prompt, response[1]['content'])
                row_rewards.append(reward)
                row_critiques.append(critique)
        rewards.append(row_rewards)
        critiques.append(row_critiques)
    return rewards, critiques


def prompt_eval(
    dataset_name: str,
    hf_username: str,
    reward_function: str,
    nproc: int,
    batch_size: int,
    evaluation_mode: str,
    **kwargs
) -> None:
    """
    Evaluate prompts and their responses, calculate rewards.
    """
    # Load dataset
    dataset = load_dataset(f"{hf_username}/{dataset_name}", split="train")
    df = dataset.to_pandas()

    # Get reward function
    reward_func = get_reward_function(reward_function, evaluation_mode=evaluation_mode, **kwargs)

    # Prepare data for batch processing
    data = [{'prompt': row['prompt'], 'responses': [row[f'generate_{i}'] for i in range(5)]} for _, row in df.iterrows()]

    # Process data in batches using ThreadPoolExecutor
    all_rewards = []
    all_critiques = []
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = [executor.submit(process_batch, data[i:i+batch_size], reward_func, evaluation_mode) 
                   for i in range(0, len(data), batch_size)]
        for future in as_completed(futures):
            batch_rewards, batch_critiques = future.result()
            all_rewards.extend(batch_rewards)
            all_critiques.extend(batch_critiques)

    # Add new columns
    df['rewards'] = all_rewards
    df['critiques'] = all_critiques
    df['reward_mean'] = df['rewards'].apply(np.mean)
    df['reward_var'] = df['rewards'].apply(np.var)
    df['reward_maxmin'] = df['rewards'].apply(lambda x: max(x) - min(x))

    # Save and push to hub
    new_dataset = Dataset.from_pandas(df)
    new_dataset.push_to_hub(f"{hf_username}/{dataset_name}-re", split="train", private=True)


def prompt_sample(
    dataset_name: str,
    hf_username: str,
    metric: str,
    frac: float,
) -> None:
    """Sample prompts based on a specified metric."""
    # Load dataset
    dataset = load_dataset(f"{hf_username}/{dataset_name}-re", split="train")
    df = dataset.to_pandas()

    # Calculate weights
    weights = df[metric] / df[metric].sum()

    # Sample rows
    sampled_df = df.sample(n=int(len(df) * frac), weights=weights, replace=False)

    # Save and push to hub
    new_dataset = Dataset.from_pandas(sampled_df)
    new_dataset.push_to_hub(f"{hf_username}/{dataset_name}-re-sample-{metric}", split="train", private=True)


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
        model_path=args.hf_model_path,
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