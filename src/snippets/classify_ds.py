"""
Given a topic list and a HF dataset with prompts,
classify the prompts into topics using a pre-trained model;
add the topic as a column to the dataset and push the entire dataset (all splits) to the hub.
"""

import argparse
from datasets import load_dataset, DatasetDict
from transformers import pipeline
from tqdm import tqdm


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
        default='facebook/bart-large-mnli', 
        help="Model to use for zero-shot classification"
    )
    
    parser.add_argument(
        "--categories", 
        type=str, 
        nargs='+', 
        default=["Writing", "Reasoning", "Math", "Coding",
                 "Summarization", "STEM", "Humanities", "Roleplay"],
        help="List of topics for classification"
    )
    
    parser.add_argument(
        "--public", 
        action='store_true', 
        help="Set the output dataset to public"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the zero-shot classification model
    classifier = pipeline('zero-shot-classification', 
                          model=args.model, 
                          device_map='auto')

    # Function to classify prompts with progress tracking
    def classify_prompts(prompts):
        topics = []
        for prompt in tqdm(prompts, desc="Classifying Prompts"):
            result = classifier(prompt, candidate_labels=args.categories)
            topics.append(result['labels'][0])
        return topics

    # Load the dataset to get all splits
    dataset_dict = load_dataset(args.dataset)
    updated_dataset_dict = DatasetDict()

    # Iterate over each split in the dataset
    for split in dataset_dict.keys():
        ds = dataset_dict[split]
        
        # Extract prompts and classify them
        prompts = ds['prompt']
        topics = classify_prompts(prompts)
        
        # Add the topic column to the dataset
        ds = ds.add_column('topic', topics)
        
        # Add the updated split to the new DatasetDict
        updated_dataset_dict[split] = ds

    # Push the entire updated dataset (all splits) to the hub
    updated_dataset_dict.push_to_hub(
        args.output_dataset if args.output_dataset else args.dataset, 
        private=not args.public)


if __name__ == "__main__":
    main()
