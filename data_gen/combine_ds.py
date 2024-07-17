"""
A simple helper function to shuffle and combine datasets.
Only prompts are retained.
"""

import argparse
from datasets import load_dataset, concatenate_datasets


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Allow for multiple datasets as input
    parser.add_argument(
        "--datasets", 
        type=str, 
        nargs='+', 
        default=[
            'cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards',
            'cat-searcher/responses-gemma-1.1-2b-it-split-0-all-hf-rewards-resample-evol'
        ],
        help="List of datasets to combine"
    )

    parser.add_argument(
        "--output_dataset", 
        type=str, 
        default='cat-searcher/responses-gemma-1.1-2b-it-split-0-evol-mixed',
        help="Name of the output dataset"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load and process datasets
    datasets = []
    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name, split='train')
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != 'prompt'])
        datasets.append(dataset)

    # Combine the datasets
    combined_dataset = concatenate_datasets(datasets)

    # Shuffle the combined dataset
    shuffled_dataset = combined_dataset.shuffle(seed=42)

    # Push the combined and shuffled dataset to the hub
    shuffled_dataset.push_to_hub(args.output_dataset, split='train', private=True)


if __name__ == "__main__":
    main()
