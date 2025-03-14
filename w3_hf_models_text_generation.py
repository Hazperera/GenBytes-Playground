import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig

# Load environment variables (if needed)
load_dotenv(override=True)

# Retrieve API key
HUGGINGFACE_KEY = os.getenv("HF_TOKEN")
print(f"API Key Loaded: {bool(HUGGINGFACE_KEY)}")

# Define model names
MODELS = {
    "LLAMA": "meta-llama/Meta-Llama-3.1-8B",
    "PHI3": "microsoft/Phi-3-mini-4k-instruct",
    "GEMMA2": "google/gemma-2-2b-it",
    "QWEN2": "Qwen/Qwen2-7B-Instruct",
    "MIXTRAL": "mistralai/Mixtral-8x7B-Instruct-v0."
}

# Set device for computation (CPU for laptops)
device = "cpu"

# Define quantization configuration
quant_config = BitsAndBytesConfig(load_in_8bit=True)


def generate_text(model_key, messages):
    if model_key not in MODELS:
        print(f"Model {model_key} not found!")
        return

    model_name = MODELS[model_key]
    print(f"\nGenerating output using {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages,
                                           return_tensors="pt",
                                           add_generation_prompt=True).to(device
                                                                          )
    streamer = TextStreamer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map=device,
                                                 quantization_config=quant_config
                                                 )
    model.generate(inputs, max_new_tokens=80, streamer=streamer)

    del tokenizer, streamer, model, inputs
    torch.cuda.empty_cache()


# Define prompts for each model
PROMPTS = {
    "PHI3": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    ],
    "GEMMA2": [
        {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
    ],
    "LLAMA": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What are some recent advancements in AI?"}
    ],
    "QWEN2": [
        {"role": "system", "content": "You are a knowledgeable AI"},
        {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "MIXTRAL": [
        {"role": "system", "content": "You are an AI specialized in science and technology"},
        {"role": "user", "content": "Describe the future of renewable energy."}
    ]
}


# Run text generation for all models
def run_all_models():
    for model_key, messages in PROMPTS.items():
        generate_text(model_key, messages)


# Execute the function to run all models
run_all_models()