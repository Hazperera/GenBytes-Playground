import os
import json
import base64
import gradio as gr
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import openai

# Initialization
load_dotenv(override=True)

# OpenAI API Key
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

# OpenAI Client
openai_client = OpenAI()
MODEL = "gpt-4o-mini"

# System message
system_message = "You are a helpful assistant for an Airline called FlightFinder. "
system_message += "Give short, courteous answers, no more than 1 sentence. "
system_message += "Always be accurate. If you don't know the answer, say so."

# Ticket price function
ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

# Tool function
def get_ticket_price(destination_city):
    print(f"Tool get_ticket_price called for {destination_city}")
    city = destination_city.lower()
    return ticket_prices.get(city, "Unknown")

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city. Call this whenever you need to know the ticket price, for example when a customer asks 'How much is a ticket to this city'",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": price_function}]


def chat(history):
    """Chat with the AI Assistant"""
    messages = [{"role": "system", "content": system_message}] + history
    response = openai_client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
    image = None

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        response, city = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        image = artist(city) if city else None
        response = openai_client.chat.completions.create(model=MODEL, messages=messages)

    reply = response.choices[0].message.content
    new_history = history + [{"role": "assistant", "content": reply}]
    talker(reply)
    return new_history, image


def handle_tool_call(message):
    """Handle tool call"""
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    city = arguments.get('destination_city')

    if not city:
        return {"role": "tool", "content": json.dumps({"error": "City not provided"})}, None

    price = get_ticket_price(city)
    response = {
        "role": "tool",
        "content": json.dumps({"destination_city": city, "price": price}),
        "tool_call_id": tool_call.id
    }
    return response, city


def artist(city):
    """Generate an image of a vacation in the city"""
    image_response = openai_client.images.generate(
            model="dall-e-3",
            prompt=f"An image representing a vacation in {city}, showing tourist spots and everything unique about {city}, in a vibrant pop-art style",
            size="1024x1024",
            n=1,
            response_format="b64_json",
        )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def talker(message):
    """Generate audio from text"""
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input=message
    )
    audio_stream = BytesIO(response.content)
    audio = AudioSegment.from_file(audio_stream, format="mp3")
    play(audio)


def transcribe_audio(audio_file):
    """Transcribe audio file"""
    with open(audio_file, "rb") as file:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=file
        )
    return transcript.text

# User Interface
with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Upload an audio file for transcription")
    with gr.Row():
        clear = gr.Button("Clear")

    def do_entry(message, history):
        new_history = history + [{"role": "user", "content": message}]
        return "", new_history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        chat, inputs=[chatbot], outputs=[chatbot, image_output]
    )
    audio_input.change(transcribe_audio, inputs=[audio_input], outputs=[entry])
    clear.click(lambda: ("", None, None), inputs=None, outputs=[entry, chatbot, image_output], queue=False)

# Launch the UI
ui.launch(inbrowser=True)
