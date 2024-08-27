"""
Given a topic list and a HF dataset with prompts,
classify the prompts into topics using a pre-trained model;
add the topic as a column to the dataset and push to the hub.
"""

import argparse
from datasets import load_dataset
from transformers import pipeline
from collections import Counter
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
        default="cat-searcher/ultrafeedback-dpo-gemma-2-9b-it-split-1-iter-1-subset-reward_gap-0.25", 
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
        "--split", 
        type=str, 
        default='train', 
        help="Dataset split to process"
    )
    
    parser.add_argument(
        "--private", 
        action='store_false', 
        help="Set the output dataset to private"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Load the dataset from Hugging Face
    ds = load_dataset(args.dataset, split=args.split)

    # Extract prompts
    prompts = ds['prompt']

    # Load zero-shot classification model
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

    # Apply classification to all prompts
    topics = classify_prompts(prompts)
    ds = ds.add_column('topic', topics)
    
    # Get the frequency
    topic_counts = Counter(topics)
    total_prompts = len(topics)
    topic_freqs = [topic_counts[topic] / total_prompts for topic in topics]
    ds = ds.add_column('topic_freq', topic_freqs)
    
    # Push the updated dataset to the hub
    ds.push_to_hub(args.output_dataset if args.output_dataset else args.dataset, 
                   private=args.private, 
                   split=args.split)


if __name__ == "__main__":
    main()
