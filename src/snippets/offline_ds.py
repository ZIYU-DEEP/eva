from datasets import load_dataset, DatasetDict

# Load the raw dataset
dataset_name = 'snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset'
raw_dataset = load_dataset(dataset_name)

# Function to create and push datasets
def push_datasets(raw_dataset, base_name, n_splits):
    for i in range(1, n_splits + 1):
        train_split = f'train_iteration_{i}'
        test_split = f'test_iteration_{i}'
        
        # Create a new dataset dictionary with train and test splits
        new_dataset = DatasetDict({
            'train': raw_dataset[train_split],
            'test': raw_dataset[test_split]
        })
        
        # Dataset name for this split
        new_dataset_name = f'{base_name}-{i}'
        
        # Push to hub
        new_dataset.push_to_hub(new_dataset_name, private=True)
        print(f"Pushed {new_dataset_name} to Hugging Face Hub.")

# Define the base name and number of splits
base_name = 'cat-searcher/ultra-feedback-split'
n_splits = 3

# Push the datasets
push_datasets(raw_dataset, base_name, n_splits)
