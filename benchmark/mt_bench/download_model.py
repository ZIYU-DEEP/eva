import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return tokenizer, model

def main():
    parser = argparse.ArgumentParser(description='Load a transformer model.')
    parser.add_argument('--model-path', type=str, help='The name of the model to load')
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)
    print(f"Model {args.model_path} loaded successfully.")

if __name__ == "__main__":
    main()
