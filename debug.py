# debugging_script.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def main():
    # Load the model and tokenizer
    model_name = 'google/gemma-2-9b-it'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Ensure special tokens are set correctly (should be handled by the tokenizer)
    # But we can double-check
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or '<pad>'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare a conversation using the chat template
    messages = [
        {"role": "user", "content": "Can you help me debug on this?"},
    ]

    # Apply the chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    # Set up generation configuration
    generation_config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.7,
        top_k=None,  # Set to None to disable top-k filtering properly
        top_p=0.9,
        do_sample=True,
        use_cache=True,
    )

    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            generation_config=generation_config,
        )

    # Decode and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()
