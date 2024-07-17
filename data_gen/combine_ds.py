"""
A simple helper function to shuffle and combine datasets.
Only prompts are retained.
"""

import argparse
from datasets import load_dataset, Dataset
from datasets import concatenate_datasets


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Just temp things
    # Better write it as a list of datasets
    parser.add_argument(
        "--dataset_1", 
        type=str, 
        default='cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards')

    parser.add_argument(
        "--dataset_2", 
        type=str, 
        default='cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards-resample-evol')

    parser.add_argument(
        "--output_dataset", 
        type=str, 
        default='cat-searcher/responses-gemma-1.1-2b-it-split-0-evol-mixed')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load datasets
    dataset_1 = load_dataset(args.dataset_1, split='train')
    dataset_2 = load_dataset(args.dataset_2, split='train')

    # Retain only the 'prompt' column
    dataset_1 = dataset_1.remove_columns([col for col in dataset_1.column_names if col != 'prompt'])
    dataset_2 = dataset_2.remove_columns([col for col in dataset_2.column_names if col != 'prompt'])

    # Combine the datasets
    combined_dataset = concatenate_datasets([dataset_1, dataset_2])

    # Shuffle the combined dataset
    shuffled_dataset = combined_dataset.shuffle(seed=42)

    # Push the combined and shuffled dataset to the hub
    shuffled_dataset.push_to_hub(args.output_dataset, split='train', private=True)


if __name__ == "__main__":
    main()
