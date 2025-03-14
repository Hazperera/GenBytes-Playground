import os
from openai import OpenAI
import ollama
from dotenv import load_dotenv


# Load environment variables (if needed)
load_dotenv(override=True)

# Retrieve API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"API Key Loaded: {bool(OPENAI_API_KEY)}")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Model references
MODEL_GPT = "gpt-4o-mini"
MODEL_LLAMA = "llama3.2"

# System prompt
SYSTEM_PROMPT = """
You are a friendly and knowledgeable math tutor designed to help Advanced Level students not only understand mathematical concepts but also develop the ability to teach them to others.

You guide students through logical reasoning and problem-solving, encouraging them to think like detectives solving a mystery. Instead of simply providing answers, ask thought-provoking questions that lead them toward discovering solutions on their own—just as Sherlock Holmes pieces together clues to solve a case.

When explaining complex topics, use crime mystery or detective-related analogies to make the concepts engaging and memorable. For example, describe calculus as "analyzing the footprints left behind by a suspect" (derivatives) or probability as "figuring out the most likely suspect based on the evidence."

Encourage students to break down concepts as if they were detectives explaining their findings to a jury. If they struggle, provide hints, clues, or guiding questions, just as a great detective would help their apprentice see the bigger picture.

If a student asks for examples, provide them only upon request. Additionally, encourage students to create their own detective-style explanations and explain their reasoning.

Maintain a patient, supportive, and encouraging tone, empowering students to develop confidence not just in solving math problems, but in explaining them like a master detective solving a case.
"""

# User prompt
USER_PROMPT = """
Explain the concept of inference.
"""


def stream_answer(user_prompt: str) -> str:
    """
    Generates a concise explanation for the given code using OpenAI API.

    Args:
        user_prompt (str): The user input to be explained.

    Returns:
        str: The generated explanation or an error message.
    """
    try:
        stream = client.chat.completions.create(
            model=MODEL_GPT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )

        response = ""

        for chunk in stream:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                response += delta.content

        return response

    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "Error: Unable to retrieve explanation."


def ollama_answer(user_prompt: str) -> str:
    """
    Generates a concise explanation for the given code using Ollama API.

    Args:
        user_prompt (str): The user input to be explained.

    Returns:
        str: The generated explanation or an error message.
    """
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        response = ollama.chat(model=MODEL_LLAMA, messages=messages)
        return response.get("message", {}).get("content", "Error: No response received.")

    except Exception as e:
        print(f"Ollama API Error: {e}")
        return "Error: Unable to retrieve explanation."


if __name__ == "__main__":
    print("---- OpenAI Streaming Explanation ----")
    openai_response = stream_answer(USER_PROMPT)
    print(f"\nFinal Response:\n{openai_response}")

    print("\n---- Ollama Explanation ----")
    ollama_response = ollama_answer(USER_PROMPT)
    print(f"\nFinal Response:\n{ollama_response}")