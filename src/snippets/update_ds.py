"""
Given an input dataset with a specific column and a target dataset with prompts,
assert that the prompts are identical, add the specified column from the input dataset
to the target dataset, and push the updated target dataset to the hub.
"""

import argparse
from datasets import load_dataset
import numpy as np


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Dataset arguments
    parser.add_argument(
        "--input_dataset", 
        type=str, 
        default="cat-searcher/ultrafeedback-gemma-split-1", 
        help="Hugging Face input dataset with the column to add"
    )
    
    parser.add_argument(
        "--target_dataset", 
        type=str, 
        default="cat-searcher/ultrafeedback-dpo-gemma-2-9b-it-split-1-iter-1-all-hf-rewards", 
        help="Hugging Face target dataset to update with the column"
    )
    
    parser.add_argument(
        "--output_dataset", 
        type=str, 
        default='',
        help="Leave blank if the same as the target dataset."
    )
    
    parser.add_argument(
        "--column_to_add", 
        type=str, 
        default="topic", 
        help="Name of the column to add from the input dataset to the target dataset"
    )
    
    parser.add_argument(
        "--public", 
        action='store_true', 
        help="Set the output dataset to private"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the input and target datasets to get all splits
    input_dataset_dict = load_dataset(args.input_dataset)
    target_dataset_dict = load_dataset(args.target_dataset)
    
    # Iterate over each split in the target dataset
    for split in target_dataset_dict.keys():
        input_ds = input_dataset_dict[split]
        target_ds = target_dataset_dict[split]
        
        # Assert that the prompt columns are identical
        assert np.array_equal(input_ds['prompt'], target_ds['prompt']), \
            f"Prompt columns in the {split} split do not match between input and target datasets."
        
        # Add the specified column from input dataset to target dataset
        target_ds = target_ds.add_column(args.column_to_add, input_ds[args.column_to_add])
        
        # Push the updated target dataset split to the hub
        target_ds.push_to_hub(args.output_dataset if args.output_dataset else args.target_dataset, 
                              private=not args.public, 
                              split=split)


if __name__ == "__main__":
    main()
