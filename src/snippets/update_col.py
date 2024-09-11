from datasets import load_dataset, Dataset
import pandas as pd
import ast  # For safely evaluating strings to list

# Load the dataset from Hugging Face
dataset = load_dataset("cat-searcher/ultrafeedback-dpo-gemma-2-9b-it-split-1-iter-1-all-hf-rewards")

# Convert to a pandas DataFrame
df = pd.DataFrame(dataset['train'])  # Adjust 'train' if using a different split

# Function to calculate the reward doublet gap
def reward_doublet_gap(rewards):
    if isinstance(rewards, str):
        rewards = ast.literal_eval(rewards)  # Safely convert string to list
    rewards = [float(r) for r in rewards]  # Convert to floats
    sorted_rewards = sorted(rewards, reverse=True)
    return sorted_rewards[0] - sorted_rewards[1]

# Apply the function to create a new column 'reward_dts'
df['reward_dts'] = df['rewards'].apply(reward_doublet_gap)

# Convert the DataFrame back to a Hugging Face Dataset
new_dataset = Dataset.from_pandas(df)

# Push the updated dataset to Hugging Face Hub
new_dataset.push_to_hub("cat-searcher/ultrafeedback-dpo-gemma-2-9b-it-split-1-iter-1-all-hf-rewards")
