"""
Given results from RankRM on generated responses,
prepare the datasets in the format of dataframes for training,
and push to the hub.
"""

import numpy as np
from datasets import load_dataset, Dataset
from pathlib import Path
import json
import math
import argparse
import pandas as pd
import datasets
import os
import numpy as np
import os
import pandas as pd
import datasets


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir", type=str, 
                        default="responses-gemma-1.1-2b-it-split-0")
    parser.add_argument("--dataset_name", type=str, 
                        default="cat-searcher/ultra-feedback-split-0")
    parser.add_argument("--data_root", type=str, 
                        default="./data")
    parser.add_argument("--n_pairs", type=int, default=5)
    parser.add_argument("--n_gpus", type=int, default=8)
    parser.add_argument("--hf_username", type=str, default="cat-searcher")
    return parser.parse_args()


def push_dataset(
    data_path: str='./data/ranking/responses-gemma-1.1-2b-it-split-0', 
    hf_username: str='cat-searcher',
    hf_filename: str='responses-gemma-1.1-2b-it-split-0-all',
    ):
    """
    Push the dataset to the hub.
    """
    
    # Set the data path
    path_train = str(Path(data_path) / 'train.parquet')
    path_test = str(Path(data_path) / 'test.parquet')
    
    # Prepare the data
    train = Dataset.from_parquet(path_train)
    try:
        test = Dataset.from_parquet(path_test)
    except:
        # Temporary solution to make the code run
        # Cannot use for test/evaluation purpose
        train_df = pd.read_parquet(path_train)
        test_df = train_df.sample(n=500)
        test_df.to_parquet(path_test, index=False)
        test = Dataset.from_parquet(path_test)
        
    train.push_to_hub(f"{hf_username}/{hf_filename}", split="train", private=True)
    test.push_to_hub(f"{hf_username}/{hf_filename}", split="test", private=True)
    
    return None


def from_ranks(
    dataset_name: str = "cat-searcher/ultra-feedback-split-0",
    output_dir: str = "gemma-split-0",
    gen_dir: str = "./data/generated/gemma-split-0",
    ranking_dir: str = "./data/ranking/gemma-split-0",
    n_gpus: str = 0,
    n_pairs: str = 5,
    hf_username: str = 'cat-searcher',
    ):
    """
    Take the numpy array, generate the df with columns below:
        - prompt_id
        - prompt
        - generate_{i}
        - probability: list of lists
        - rm_scores: list of lists
    """
    # Load the data
    data = load_dataset(dataset_name, split="train")
    print(f"Length of dataset: {len(data)}")

    # Initialize the scores, probs, and rm_scores
    scores = [0 for _ in range(len(data))]
    probs = []
    rm_scores = []
    
    # Fill the scores from the numpy array from ranking scores
    for idx in range(n_gpus):
        locals = np.load(Path(ranking_dir) / f'{idx}_{idx}.npy')
        locals = list(locals)
        for lidx, sc in enumerate(locals):
            frac_len = math.ceil(len(data) / n_gpus)
            scores[idx * frac_len + lidx] = sc

    # Generate probs and rm_scores for each pair
    for idx, score in enumerate(scores):
        prb = np.zeros((n_pairs, n_pairs))
        for i in range(n_pairs):
            for j in range(n_pairs):
                # This looks to be problematic
                # See https://github.com/uclaml/SPPO/issues/15
                prb[i][j] = 1 / (1 + np.exp(score[j] - score[i]))
        prb = prb.tolist()
        probs.append(prb)
        rm_scores.append(score)

    print("Saving probabilities...")
    with open(Path(gen_dir) / 'probabilities.json', 'w') as f:
        json.dump(probs, f)

    df = data.to_pandas()
    for i in range(n_pairs):
        with open(Path(gen_dir) / f'responses_{i}.json') as f:
            responses = json.load(f)
        formatted_responses = [
            [
                {"content": data[j]["prompt"], 
                 "role": "user"},
                
                {"content": responses[j], 
                 "role": "assistant"},
            ]
            for j in range(len(data))
        ]
        df[f'generate_{i}'] = formatted_responses

    df['probability'] = probs
    df['rm_scores'] = rm_scores
    
    # Save the data into the format of a parquet file
    df.to_parquet(str(Path(gen_dir) / 'train.parquet'))
    
    # Push to huggingface
    push_dataset(
        data_path=gen_dir,
        hf_username=hf_username,
        hf_filename=f'{str(output_dir)}-all',
    )
    
    return None

def prepare_score(
    output_dir: str = "gemma-split-0",
    gen_dir: str = "./data/generated/gemma-split-0",
    ranking_dir: str = "./data/ranking/gemma-split-0",
    hf_username: str = 'cat-searcher',
    ):
    """
    Take the previous df, generate a new df with the columns below:
        - chosen
        - rejected
        - chosen_prob
        - chosen_probs_win
        - chosen_probs_los
    """
    
    # Load dataset and convert to DataFrame
    train = datasets.load_dataset(str(gen_dir))
    train = pd.DataFrame(train['train'])

    # Calculate metrics and probabilities
    metrics = train['rm_scores'].apply(lambda x: np.array(x[-5:]))
    metrics_prob = train['probability'].apply(lambda x: np.stack(x).sum(axis=1))
    maxmin = metrics.apply(lambda x: [x.argmax(), x.argmin()])

    # Reorganize the DataFrame for easy access
    train_ordered = train[['generate_0', 
                           'generate_1', 
                           'generate_2', 
                           'generate_3', 
                           'generate_4', 
                           'probability']]

    # Determine chosen and rejected items based on maxmin indices
    chosen = [train_ordered.iloc[i, maxmin[i][0]] 
              for i in range(len(train_ordered))]
    rejected = [train_ordered.iloc[i, maxmin[i][1]] 
                for i in range(len(train_ordered))]

    # Calculate probabilities for chosen and rejected items
    chosen_probs = [train_ordered['probability'].iloc[i][maxmin[i][0]][maxmin[i][1]] 
                    for i in range(len(train_ordered))]
    chosen_probs_win = [metrics_prob[i][maxmin[i][0]] / len(metrics_prob.iloc[0]) 
                        for i in range(len(metrics_prob))]
    chosen_probs_lose = [metrics_prob[i][maxmin[i][1]] / len(metrics_prob.iloc[0]) 
                         for i in range(len(metrics_prob))]

    # Create a new DataFrame with the results
    train_new = pd.DataFrame({
        'chosen': chosen,
        'rejected': rejected,
        'chosen_probs': chosen_probs,
        'chosen_probs_win': chosen_probs_win,
        'chosen_probs_lose': chosen_probs_lose
    })

    # Save train and test datasets to parquet files
    train_new.to_parquet(str(Path(ranking_dir) / 'train.parquet'), index=False)
    print(f'Saved file to {Path(ranking_dir)}/train.parquet.')

    # Temporary solution to make the code run, cannot use for test/evaluation purpose
    test = train_new.sample(n=500)
    test.to_parquet(str(Path(ranking_dir) / 'test.parquet'), index=False)
    
    # Push to huggingface
    push_dataset(
        data_path=ranking_dir,
        hf_username=hf_username,
        hf_filename=f'{str(output_dir)}-pair',
    )

    return None


def main():
    # -------------- Set up the arguments --------------- #
    args = parse_arguments()
    
    dataset_name = args.dataset_name
    n_gpus = args.n_gpus
    n_pairs = args.n_pairs
    hf_username = args.hf_username

    output_dir = Path(args.output_dir)  # Shared across files
    data_root = Path(args.data_root)
    gen_dir = data_root / 'generated' / output_dir
    ranking_dir = data_root / 'ranking' / output_dir
    
    
    # -------------- Generate the dataset --------------- #
    # Generate datasets for all responses
    from_ranks(
        dataset_name=dataset_name,
        output_dir=output_dir,
        gen_dir=gen_dir,
        ranking_dir=ranking_dir,
        n_gpus=n_gpus,
        n_pairs=n_pairs,
        hf_username=hf_username,
    )

    # Get the training dataset with the chosen and rejected pair
    prepare_score(
        output_dir=output_dir,
        gen_dir=gen_dir,
        ranking_dir=ranking_dir,
        hf_username=hf_username,
    )

    
if __name__ == "__main__":
    main()
