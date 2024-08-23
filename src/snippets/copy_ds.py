"""
A simple helper function to copy and push datasets.
"""

import argparse
from datasets import load_dataset


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Copy and push a dataset to a new name on the Hugging Face Hub.")
    
    parser.add_argument(
        "--source_dataset", 
        type=str, 
        help="The name of the source dataset to copy",
        default="cat-searcher/ultrafeedback-dpo-gemma-2-9b-it-split-1-iter-0-pair",
    )
    
    parser.add_argument(
        "--target_dataset", 
        type=str, 
        help="The name for the new copied dataset",
        default="cat-searcher/ultrafeedback-rpo-gemma-2-9b-it-split-1-iter-0-pair",
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the source dataset with all splits
    dataset_dict = load_dataset(args.source_dataset)

    # Push each split of the dataset to the hub under the new name
    for split, dataset in dataset_dict.items():
        dataset.push_to_hub(args.target_dataset, 
                            split=split, 
                            private=True)

if __name__ == "__main__":
    main()
