# models/qwen_niche_selector.py

import os
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
import torch

# Define custom model path inside your project folder
local_model_dir = os.path.join(
    "C:/Users/T5M/OneDrive/Desktop/PROJECT DOPAMINE/know_it_all_agent", "qwen_Qwen3_0_6B"
)

# Download and cache model locally only if not already present
if not os.path.exists(local_model_dir):
    local_model_dir = snapshot_download(
        "qwen/Qwen3-0.6B",
        cache_dir=local_model_dir
    )

# Load Qwen model and tokenizer from custom path
tokenizer = AutoTokenizer.from_pretrained(local_model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_dir, trust_remote_code=True).to("cpu")

def generate_qwen_response(prompt: str, max_tokens: int = 256) -> str:
    """Generates a response from Qwen model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_niches(niche_list, base_prompt_path="know_it_all_agent/prompts/base_prompt.txt"):
    """Evaluates a list of niches and returns scored responses."""
    with open(base_prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    results = []
    for niche in niche_list:
        prompt = prompt_template.replace("{NICHE}", niche)
        response = generate_qwen_response(prompt)
        results.append({"niche": niche, "response": response})
    return results

# Test
if __name__ == "__main__":
    test_niches = ["personal finance", "AI tools review", "remote work productivity"]
    for result in evaluate_niches(test_niches):
        print(f"\n🧠 Niche: {result['niche']}\n{result['response']}\n")
