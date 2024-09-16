# debugging_script.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def main():
    # Load the model and tokenizer
    model_name = 'google/gemma-2-9b-it'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Ensure special tokens are set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or '<pad>'
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    else:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.bos_token_id is None:
        model.config.bos_token_id = tokenizer.bos_token_id

    # Prepare a simple Italian input
    input_text = "Please translate this into English: Ciao, come stai?"
    inputs = tokenizer(input_text, return_tensors='pt').to(device)

    # Set up generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.7,
        top_k=None,  # Set to None to disable top-k filtering properly
        top_p=0.9,
        do_sample=True,
        use_cache=True,
    )

    # Run generation step by step to debug
    with torch.no_grad():
        # Step 1: Get logits
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, seq_length, vocab_size)

        # Check for NaNs or Infs in logits
        if torch.isnan(logits).any():
            print("NaN detected in logits")
        if torch.isinf(logits).any():
            print("Inf detected in logits")

        # Get the logits of the last token
        last_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        # Apply temperature scaling
        scaled_logits = last_token_logits / generation_config.temperature

        # Check for NaNs or Infs after scaling
        if torch.isnan(scaled_logits).any():
            print("NaN detected in scaled logits")
        if torch.isinf(scaled_logits).any():
            print("Inf detected in scaled logits")

        # Apply top_p nucleus sampling
        sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = probs.cumsum(dim=-1)

        # Remove tokens with cumulative probability above the threshold (top_p)
        sorted_indices_to_remove = cumulative_probs > generation_config.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Set logits of removed tokens to -inf
        sorted_logits[sorted_indices_to_remove] = float('-inf')

        # Map sorted logits back to original indices
        unsorted_logits = torch.empty_like(scaled_logits).scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        # Apply softmax to get probabilities
        probs = torch.softmax(unsorted_logits, dim=-1)

        # Check for NaNs, Infs, or negative values in probabilities
        if torch.isnan(probs).any():
            print("NaN detected in probabilities")
        if torch.isinf(probs).any():
            print("Inf detected in probabilities")
        if (probs < 0).any():
            print("Negative values detected in probabilities")
        if not torch.isclose(probs.sum(dim=-1), torch.tensor(1.0).to(device)).all():
            print("Probabilities do not sum to 1")
            print("Probabilities sum to:", probs.sum(dim=-1))

        # Sample from the distribution
        try:
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        except RuntimeError as e:
            print("Error during sampling:", e)
            return

        # Append the sampled token to the input_ids
        generated_ids = torch.cat([inputs['input_ids'], next_token.unsqueeze(-1)], dim=-1)

        # Decode and print the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Generated text:", generated_text)


if __name__ == "__main__":
    main()
