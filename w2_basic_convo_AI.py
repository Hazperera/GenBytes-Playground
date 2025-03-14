import os
from dotenv import load_dotenv
from openai import OpenAI
import ollama

# Load environment variables
load_dotenv(override=True)

api_keys = {
    "OpenAI": os.getenv('OPENAI_API_KEY'),
    "Anthropic": os.getenv('ANTHROPIC_API_KEY'),
    "Google": os.getenv('GOOGLE_API_KEY'),
    "DeepSeek": os.getenv('DEEPSEEK_API_KEY')
}

for name, key in api_keys.items():
    if key:
        print(f"{name} API Key exists and begins {key[:8]}")
        break
else:
    print("API Key not set")

# Initialize OpenAI client
openai = OpenAI(api_key=api_keys["OpenAI"])

def build_messages(system_message, conversation):
    messages = [{"role": "system", "content": system_message}]
    for turn in conversation:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    return messages

def call_gpt(conversation):
    messages = build_messages(
        "You are a chatbot who is very argumentative; you challenge everything in a snarky way.",
        conversation
    )
    completion = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
    return completion.choices[0].message.content

def call_llama(conversation):
    messages = build_messages(
        "You are a very polite chatbot, always finding common ground.",
        conversation
    )
    response = ollama.chat(model="llama3.2:latest", messages=messages)
    return response["message"]["content"]

conversation = [{"user": "Hi there", "assistant": "Hi"}]

print("Initial Conversation:")
for turn in conversation:
    print(f"User: {turn['user']}")
    print(f"Assistant: {turn['assistant']}\n")

for _ in range(5):
    gpt_next = call_gpt(conversation)
    print(f"GPT:\n{gpt_next}\n")

    llama_next = call_llama(conversation)
    print(f"Llama:\n{llama_next}\n")

    conversation.append({"user": gpt_next, "assistant": llama_next})