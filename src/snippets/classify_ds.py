"""
Given a topic list and a HF dataset with prompts,
classify the prompts into topics using a pre-trained model;
add the topic as a column to the dataset and push the entire dataset (all splits) to the hub.
"""

import argparse
import time
import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm
import openai
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor

API_MAX_RETRY = 5
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = None
MAX_WORKERS = 100  


def add_or_replace_column(ds: Dataset, column_name: str, new_column):
    if column_name in ds.column_names:
        ds = ds.remove_columns([column_name])
    ds = ds.add_column(column_name, new_column)
    return ds


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Dataset and model arguments
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="cat-searcher/ultrafeedback-gemma-split-1", 
        help="Hugging Face dataset to classify"
    )
    
    parser.add_argument(
        "--output_dataset", 
        type=str, 
        default='',
        help="Leave blank if the same as the input dataset."
    )

    parser.add_argument(
        "--model", 
        type=str, 
        default='gpt-4o-mini', 
        help="ChatGPT model to use for classification"
    )
    
    parser.add_argument(
        "--categories", 
        type=str, 
        nargs='+', 
        default=["writing", "reasoning", "math", "coding", "summarization",
                 "stem/science", "humanities", "multilingual", "other"],
        help="List of topics for classification"
    )
    
    parser.add_argument(
        "--public", 
        action='store_true', 
        help="Set the output dataset to public"
    )
    
    return parser.parse_args()

def classify_prompt_with_chatgpt(prompt, categories, model, client):
    messages = [
        {"role": "system", 
         "content": "You are a helpful assistant that classifies text into predefined categories."},
        {"role": "user", "content": f"Classify the following prompt into one of these categories (answer only in lowercase with one of the category name in the list, without any punctuation. DO NOT ADD ANYTHING ELSE): {categories}\n\nPROMPT: {prompt}\n\nCATEGORY:"}
    ]

    output = API_ERROR_OUTPUT
    for _ in range(API_MAX_RETRY):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                n=1,
                temperature=0.0,
                max_tokens=4096  # Max token length for prompt + response
            )
            raw_output = response.choices[0].message.content.strip().lower()
            output = raw_output if raw_output in categories else "other"
            break
        except openai.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            time.sleep(API_RETRY_SLEEP) 
        except openai.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            time.sleep(API_RETRY_SLEEP)
        except openai.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(API_RETRY_SLEEP)
    return output


def classify_chunk(prompts_chunk, categories, model, client):
    topics = []
    for prompt in tqdm(prompts_chunk, desc="Classifying Prompts"):
        topic = classify_prompt_with_chatgpt(prompt, categories, model, client)
        topics.append(topic)
    return topics


def main():
    # Parse command line arguments
    args = parse_arguments()

    client = OpenAI()

    # Load the dataset to get all splits
    dataset_dict = load_dataset(args.dataset)
    updated_dataset_dict = DatasetDict()

    # Iterate over each split in the dataset
    for split in dataset_dict.keys():
        ds = dataset_dict[split]
        
        # Extract prompts and classify them using ChatGPT with parallel processing
        prompts = ds['prompt']
        chunk_size = len(prompts) // MAX_WORKERS

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_chunk = {
                executor.submit(classify_chunk, prompts[i:i + chunk_size], args.categories, args.model, client): i
                for i in range(0, len(prompts), chunk_size)
            }

            topics = []
            for future in tqdm(future_to_chunk.keys(), desc="Collecting Results"):
                topics.extend(future.result())
        
        # Add the topic column to the dataset
        np.save('temp_topics.npy', np.array(topics))
        ds = add_or_replace_column(ds, 'topic', topics)
        
        # Add the updated split to the new DatasetDict
        updated_dataset_dict[split] = ds

    # Push the entire updated dataset (all splits) to the hub
    updated_dataset_dict.push_to_hub(
        args.output_dataset if args.output_dataset else args.dataset, 
        private=not args.public)


if __name__ == "__main__":
    main()
