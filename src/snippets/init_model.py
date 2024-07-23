"""
So that it is easier to keep the naming fashion.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = 'google/gemma-2-9b-it'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define the new repository name
new_repo_name = 'cat-searcher/gemma-2-9b-it-sppo-iter-0'

# Push the model and tokenizer to the new repository
model.push_to_hub(new_repo_name, private=True)
tokenizer.push_to_hub(new_repo_name, private=True)

print(f"Model and tokenizer pushed to {new_repo_name} on Hugging Face Hub.")
