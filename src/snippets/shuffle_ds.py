from datasets import load_dataset, DatasetDict, concatenate_datasets


# Load the raw dataset
dataset_name = 'snorkelai/Snorkel-Mistral-PairRM-DPO-Dataset'
raw_dataset = load_dataset(dataset_name)


# Function to shuffle and split datasets
def shuffle_and_split(dataset, n_splits, seed=8964):
    dataset = dataset.shuffle(seed=seed)
    split_length = len(dataset) // n_splits
    return [
        dataset.select(range(i * split_length, (i + 1) * split_length)) 
        if i < n_splits - 1 else 
        dataset.select(range(i * split_length, len(dataset)))  # for the last split
        for i in range(n_splits)
    ]


# Concatenate all train splits and all test splits separately
train_splits = concatenate_datasets([
    raw_dataset[f'train_iteration_{i}'] for i in range(1, 4)
])
test_splits = concatenate_datasets([
    raw_dataset[f'test_iteration_{i}'] for i in range(1, 4)
])


# Shuffle and split train and test datasets into 6 parts each
train_parts = shuffle_and_split(train_splits, 6)
test_parts = shuffle_and_split(test_splits, 6)


# Function to create and push datasets
def push_datasets(train_parts, test_parts, base_name):
    for i in range(1, 7):
        # Create a new dataset dictionary with train and test splits
        new_dataset = DatasetDict({
            'train': train_parts[i - 1],
            'test': test_parts[i - 1]
        })
        
        # Dataset name for this split
        new_dataset_name = f'{base_name}-{i}'
        
        # Push to hub
        new_dataset.push_to_hub(new_dataset_name, private=True)
        print(f"Pushed {new_dataset_name} to Hugging Face Hub.")

# Define the base name
base_name = 'cat-searcher/ultrafeedback-gemma-split'

# Push the datasets
push_datasets(train_parts, test_parts, base_name)
