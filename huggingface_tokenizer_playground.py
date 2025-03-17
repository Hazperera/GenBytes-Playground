import os
from dotenv import load_dotenv
from transformers import AutoTokenizer

# Load environment variables (if needed)
load_dotenv(override=True)

# Load HuggingFace API Token
HUGGINGFACE_KEY = os.getenv("HF_TOKEN")
print(f"API Key Loaded: {bool(HUGGINGFACE_KEY)}")

# Define model names
LLAMA_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B"
PHI3_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
QWEN2_MODEL_NAME = "Qwen/Qwen2-7B-Instruct"
STARCODER2_MODEL_NAME = "bigcode/starcoder2-3b"

# Sample text for tokenization
test_text = "Let's break this sentence to show Tokenizers in action"
code_sample = """def hello_world(person): print("Hello", person)"""

# Example messages for chat templates
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
]

# Toggle models easily by setting to True/False
RUN_LLAMA = False
RUN_LLAMA_INSTRUCT = False
RUN_PHI3 = False
RUN_QWEN2 = True
RUN_STARCODER2 = False

if RUN_LLAMA:
    print("\nRunning Llama 3.1 Tokenizer...")
    llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_NAME, trust_remote_code=True)
    tokens = llama_tokenizer.encode(test_text)
    print("Tokenized Output:", tokens)
    print("Number of Tokens:", len(tokens))
    print("Decoded Output:", llama_tokenizer.decode(tokens))
    print("Added Vocabulary:", llama_tokenizer.get_added_vocab())

if RUN_LLAMA_INSTRUCT:
    print("\nRunning Llama 3.1 Instruct Tokenizer...")
    llama_instruct_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", trust_remote_code=True)
    llama_prompt = llama_instruct_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Llama Instruct Model Prompt:")
    print(llama_prompt)

if RUN_PHI3:
    print("\nRunning Phi-3 Tokenizer...")
    phi3_tokenizer = AutoTokenizer.from_pretrained(PHI3_MODEL_NAME)
    phi3_tokens = phi3_tokenizer.encode(test_text)
    print("Phi-3 Tokenized Output:", phi3_tokens)
    print("Decoded Output:", phi3_tokenizer.batch_decode(phi3_tokens))
    print("Phi-3 Chat Template:")
    print(phi3_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

if RUN_QWEN2:
    print("\nRunning Qwen2 Tokenizer...")
    qwen2_tokenizer = AutoTokenizer.from_pretrained(QWEN2_MODEL_NAME)
    qwen2_tokens = qwen2_tokenizer.encode(test_text)
    print("Qwen2 Tokenized Output:", qwen2_tokens)
    print("Decoded Output:", qwen2_tokenizer.batch_decode(qwen2_tokens))
    print("Qwen2 Chat Template:")
    print(qwen2_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

if RUN_STARCODER2:
    print("\nRunning Starcoder2 Tokenizer...")
    starcoder2_tokenizer = AutoTokenizer.from_pretrained(STARCODER2_MODEL_NAME, trust_remote_code=True)
    starcoder2_tokens = starcoder2_tokenizer.encode(code_sample)
    print("Starcoder2 Tokenized Output:")
    for token in starcoder2_tokens:
        print(f"{token} = {starcoder2_tokenizer.decode(token)}")