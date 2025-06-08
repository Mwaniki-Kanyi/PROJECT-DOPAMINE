from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Define the model name (ensure this is the correct identifier from Hugging Face Hub)
model_name = "Qwen/Qwen3-0.6B" # Or your corrected model name

# 2. Define the device to use (CPU in this case)
device = "cpu"
print(f"Using device: {device}")

# 3. Load the tokenizer and model
#    The model will be downloaded automatically if not already cached.
try:
    print(f"Loading tokenizer for {model_name}...")
    # trust_remote_code=True might be needed for some custom model architectures/tokenizers
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model {model_name} for CPU...")
    # For CPU, we specify torch_dtype=torch.float32 (default for CPU)
    # We remove device_map and will explicitly move the model to CPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32, # Use float32 for CPU
        trust_remote_code=True
        # No device_map here; we'll use .to(device)
    )
    model.to(device) # Explicitly move the model to CPU
    model.eval() # Set the model to evaluation mode (good practice for inference)

    print("Model and tokenizer loaded successfully on CPU.")

except OSError as e:
    print(f"Error loading model or tokenizer: {e}")
    print(f"Please ensure the model name '{model_name}' is correct and you have internet access.")
    print("If this is a gated model, ensure you are logged in via `huggingface-cli login`.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    exit()

# 4. Prepare your input prompt
prompt = "你好，请介绍一下你自己。" # "Hello, please introduce yourself."
print(f"\nInput prompt: {prompt}")

# 5. Tokenize the input
#    Move tokenized inputs to the same device as the model (CPU).
print("Tokenizing input...")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 6. Generate text
#    Adjust max_new_tokens and other generation parameters as needed.
#    Note: Generation on CPU will be slower than on GPU.
print("Generating text (this may take a moment on CPU)...")
try:
    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1
            # Add other generation parameters here if needed, e.g.:
            # temperature=0.7,
            # top_k=50,
            # top_p=0.95,
            # do_sample=True, # Set to True for more creative/varied outputs
        )
except Exception as e:
    print(f"Error during text generation: {e}")
    exit()

# 7. Decode and print the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated text:\n{generated_text}")

print(f"\nModel {model_name} tested successfully on CPU.")

