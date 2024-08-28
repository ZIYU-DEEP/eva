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
            'cat-searcher/ultrafeedback-split-1',
            'cat-searcher/ultrafeedback-split-2',
            'cat-searcher/ultrafeedback-split-3'
        ],
        help="List of datasets to combine"
    )
    
    parser.add_argument(
        "--ratios", 
        type=float, 
        nargs='+', 
        default=[1, 1, 1],
        help="List of ratios for each dataset"
    )

    parser.add_argument(
        "--output_dataset", 
        type=str, 
        default='cat-searcher/uf-split-1',
        help="Name of the output dataset"
    )
    
    return parser.parse_args()


# def main():
#     # Parse command line arguments
#     args = parse_arguments()

#     if len(args.datasets) != len(args.ratios):
#         raise ValueError(
#             "The number of datasets must match the number of ratios!")

#     # Load and process datasets
#     datasets = []
#     for dataset_name, ratio in zip(args.datasets, args.ratios):
#         # Load the dataset
#         dataset = load_dataset(dataset_name, split='train')
        
#         # Sample the dataset based on the given ratio
#         num_samples = int(len(dataset) * ratio)
        
#         # Shuffle the dataset
#         sampled_dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
#         # Append to the full dataset
#         datasets.append(sampled_dataset)

#     # Combine the datasets
#     combined_dataset = concatenate_datasets(datasets)

#     # Shuffle the combined dataset
#     shuffled_dataset = combined_dataset.shuffle(seed=42)

#     # Push the combined and shuffled dataset to the hub
#     shuffled_dataset.push_to_hub(args.output_dataset, 
#                                  split='train', 
#                                  private=True)


def main():
    # Parse command line arguments
    args = parse_arguments()

    if len(args.datasets) != len(args.ratios):
        raise ValueError(
            "The number of datasets must match the number of ratios!")

    combined_datasets = {}

    # Load and process datasets for each split
    for dataset_name, ratio in zip(args.datasets, args.ratios):
        # Load the dataset to get all available splits
        dataset_splits = load_dataset(dataset_name)

        for split, dataset in dataset_splits.items():
            # Sample the dataset based on the given ratio
            num_samples = int(len(dataset) * ratio)
            
            # Shuffle the dataset
            sampled_dataset = dataset.shuffle(seed=42).select(range(num_samples))
            
            # Append to the combined datasets
            if split not in combined_datasets:
                combined_datasets[split] = []
            combined_datasets[split].append(sampled_dataset)

    # Combine and shuffle the datasets for each split
    for split, datasets in combined_datasets.items():
        combined_dataset = concatenate_datasets(datasets)
        shuffled_dataset = combined_dataset.shuffle(seed=42)

        # Push the combined and shuffled dataset to the hub for this split
        shuffled_dataset.push_to_hub(args.output_dataset, 
                                     split=split, 
                                     private=True)


if __name__ == "__main__":
    main()
