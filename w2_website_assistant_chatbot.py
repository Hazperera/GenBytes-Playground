import os
import json
import uuid
from datetime import datetime
from io import BytesIO
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play

# Load environment variables
load_dotenv(override=True)

# Load API keys
api_keys = {
    "OpenAI": os.getenv("OPENAI_API_KEY"),
    "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "Google": os.getenv("GOOGLE_API_KEY"),
    "DeepSeek": os.getenv("DEEPSEEK_API_KEY"),
}

for name, key in api_keys.items():
    if key:
        print(f"{name} API Key exists and begins {key[:8]}")
        break
else:
    print("API Key not set")

# OpenAI Client
openai_client = OpenAI()
MODEL = "gpt-4o-mini"

# System message
SYSTEM_MESSAGE = (
    "You are an AI assistant for an EdTech platform that helps users find information about courses, "
    "syllabi, and learning paths. Your tone should be professional yet friendly, guiding users efficiently "
    "while keeping responses concise and clear. You do not provide opinions or personal recommendations but "
    "can suggest courses based on user needs. Always ask clarifying questions before making a recommendation. "
    "If the user asks about non-EdTech topics, politely redirect them. Never invent course details or make "
    "assumptions beyond what is provided.\n\n"
    "Now, let's start the conversation! What topics or skills are you interested in learning today?"
)

# Generate unique session ID
SESSION_ID = str(uuid.uuid4())
FEEDBACK_FILE = "feedback.json"


def save_feedback(score):
    """Saves or updates user feedback."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_data = {"session_id": SESSION_ID, "rating": score, "timestamp": timestamp}

    existing_data = []
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as file:
            try:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []

    session_found = False
    for entry in existing_data:
        if isinstance(entry, dict) and entry.get("session_id") == SESSION_ID:
            entry.update({"rating": score, "timestamp": timestamp})
            session_found = True
            break

    if not session_found:
        existing_data.append(feedback_data)

    with open(FEEDBACK_FILE, "w") as file:
        json.dump(existing_data, file, indent=4)

    print(f"Feedback updated: {score} stars at {timestamp} (Session ID: {SESSION_ID})")
    return "Thank you for your feedback!"


def chat(history):
    """Handles chatbot conversation"""
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + history
    response = openai_client.chat.completions.create(model=MODEL, messages=messages)

    reply = response.choices[0].message.content
    new_history = history + [{"role": "assistant", "content": reply}]

    talker(reply)
    return new_history, reply  # Return updated chat history and chatbot response


def talker(message):
    """Generates and plays audio from text."""
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)


def transcribe_audio(audio_file, history):
    """Transcribes audio and updates conversation"""
    with open(audio_file, "rb") as file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=file
        )

    transcribed_text = transcript.text
    user_message = {"role": "user", "content": transcribed_text}
    updated_history = history + [user_message]

    chatbot_response, reply = chat(updated_history)  # Process chat

    return transcribed_text, updated_history, chatbot_response  # Return transcription, updated history, chatbot response


# User Interface
with gr.Blocks() as ui:
    gr.Markdown("## Chat Assistant")

    chatbot = gr.Chatbot(label="Chat with AI")
    history_state = gr.State([])  # ✅ Stores conversation history
    transcription_output = gr.Textbox(label="Transcription Output", interactive=False)  # ✅ Shows transcribed text

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload an audio file for transcription")

    with gr.Row():
        clear = gr.Button("Clear")

    with gr.Row():
        feedback_component = gr.Radio(choices=[1, 2, 3, 4, 5], label="Rate your experience:")
        submit_feedback = gr.Button("Submit Feedback")
        feedback_output = gr.Textbox(label="Feedback Response")

    submit_feedback.click(save_feedback, inputs=feedback_component, outputs=feedback_output)

    # ✅ Updated: Show transcription before sending to chatbot
    audio_input.change(
        transcribe_audio,
        inputs=[audio_input, history_state],
        outputs=[transcription_output, history_state, chatbot]
    )

    # ✅ Clear button resets everything
    clear.click(lambda: ("", [], []), inputs=None, outputs=[transcription_output, history_state, chatbot])

# Launch UI
ui.launch(inbrowser=True)