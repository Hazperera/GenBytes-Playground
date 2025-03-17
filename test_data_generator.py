import os
import pandas as pd
import json
from dotenv import load_dotenv
from transformers import pipeline
from transformers import AutoModelForCausalLM

# Load environment variables (if needed)
load_dotenv(override=True)

# Retrieve API key
model_key = os.getenv("HF_TOKEN")
print(f"API Key Loaded: {bool(model_key)}")

# Define model names
MODELS = {
    "LLAMA": "meta-llama/Llama-2-7b-hf" # Smaller version of LLaMA
}

def generate_test_data(prompt, num_samples=5, model_name=MODELS["LLAMA"],device="cpu"):
        """Generates synthetic test data based on a given prompt."""
        AutoModelForCausalLM.from_pretrained(model_name)
        generator = pipeline("text-generation", model=model_name, device=0 if device == "cuda" else -1)
        results = generator(prompt, max_length=100, num_return_sequences=num_samples)
        print(f"\nGenerating output using {model_name} on {device}...")
        data = [json.loads(res["generated_text"]) for res in results]
        return pd.DataFrame(data)


# Example usage
if __name__ == "__main__":
    user_prompt = "Generate a dataset of customer profiles including name, age, email, and location"
    df = generate_test_data(user_prompt, num_samples=10)
    print(df.head())
    df.to_csv("syntest_data_customer_profiles.csv", index=False)
