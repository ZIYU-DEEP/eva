import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
device = "auto"

armo_8b = 'RLHFlow/ArmoRM-Llama3-8B-v0.1'
pairrm_8b = 'RLHFlow/pair-preference-model-LLaMA3-8B'
rm_gemma_27b = 'Skywork/Skywork-Reward-Gemma-2-27B'
rm_llama_8b = 'Skywork/Skywork-Reward-Llama-3.1-8B'

reward_model_path = rm_llama_8b

model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="flash_attention_2",
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

def hf_reward(
    prompt: str, 
    responses, 
    model, 
    tokenizer):
    
    """
    Calculate reward using a Hugging Face model.
    """

    # Reformat the input
    conversations = [
        [{"role": "user", "content": prompt},
            {"role": "assistant", "content": response}]
        for response in responses
    ]

    # Tokenize the input with padding (to process all conversations in a batch)
    input_ids = tokenizer.apply_chat_template(
        conversations, return_tensors="pt", padding=True).to(torch.device('cuda'))
    
    # Case for armo
    if "armorm-llama3-8b" in model.config._name_or_path.lower():
        with torch.no_grad():
            output = model(input_ids)
            scores = output.logits.cpu().float().tolist()

    # Case for 'Skywork-Reward-Gemma-2-27B'
    elif "skywork" in model.config._name_or_path.lower():
        with torch.no_grad():
            output = model(input_ids)
            scores = output.logits[:, 0].unsqueeze(1).cpu().float().tolist()
            
    else:
        raise NotImplementedError(
            f"Reward calculation not implemented for model: {model.config._name_or_path}")

    return list(zip(scores, [''] * len(scores)))

prompt = "What is the capital of France?"
responses = ["Paris is the capital of France.", 
             "My dear friend, the capital of France is Chicago.", 
             "I have no idea."]

result = hf_reward(
    prompt, 
    responses, 
    model, 
    tokenizer)

print(result)
breakpoint()
