import os
import gradio as gr
from openai import OpenAI
import ollama
from dotenv import load_dotenv

load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print(f"OpenAI API Key exists and begins {OPENAI_API_KEY[:8]}")
else:
    print("OpenAI API Key not set")

openai = OpenAI()

MODEL_GPT = "gpt-4o-mini"
MODEL_LLAMA = "llama3.2"

SYSTEM_PROMPT = """ 
You are a knowledgeable and engaging math tutor for advanced students, specializing in logical reasoning and problem-solving. Instead of simply providing answers, guide students through detective-style thinking, asking thought-provoking questions that lead them to discover solutions independently.  
Use crime mystery or detective analogies to make concepts memorable. For example:  
- Derivatives: "Analyzing the footprints left behind by a suspect."  
- Probability: "Determining the most likely suspect based on evidence."  
Encourage students to break down concepts as if explaining their findings to a jury. If they struggle, provide hints and guiding questions, just as a detective would with an apprentice. Offer examples only when requested and encourage students to create their own detective-style explanations.  
Maintain a patient, supportive tone, fostering confidence in both problem-solving and explaining mathematical reasoning like a master detective.
"""

def openai_stream_answer(user_prompt: str) -> str:
    try:
        response = ""
        for chunk in openai.chat.completions.create(
            model=MODEL_GPT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        ):
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
        return response
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "Error: Unable to retrieve explanation."

def ollama_stream_answer(user_prompt: str) -> str:
    try:
        response = ""
        for chunk in ollama.chat(model=MODEL_LLAMA, messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ], stream=True):
            if "message" in chunk and "content" in chunk["message"]:
                response += chunk["message"]["content"]
        return response
    except Exception as e:
        print(f"Ollama API Error: {e}")
        return "Error: Unable to retrieve explanation."

def chat_model_selector(user_prompt: str, model: str) -> str:
    if model == "OpenAI GPT":
        return openai_stream_answer(user_prompt)
    elif model == "Ollama Llama":
        return ollama_stream_answer(user_prompt)
    return "Please select a model."

gr.Interface(
    fn=chat_model_selector,
    inputs=[gr.Textbox(label="Math Question:"), gr.Radio(["OpenAI GPT", "Ollama Llama"], label="Select Model")],
    outputs=gr.Textbox(label="Answer:"),
    title="AI Math Tutor",
    flagging_mode="never"
).launch(share=True)