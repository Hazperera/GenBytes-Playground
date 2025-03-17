import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)

# Load API Keys
HUGGINGFACE_KEY = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"API Keys Loaded: HF: {bool(HUGGINGFACE_KEY)}, OpenAI: {bool(OPENAI_API_KEY)}")

# Initialize OpenAI client
openai = OpenAI(api_key=OPENAI_API_KEY)

# Constants
AUDIO_MODEL = "whisper-1"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
AUDIO_FILE_PATH = "denver_extract.mp3"


def transcribe_audio(audio_path):
    """Transcribes audio using OpenAI Whisper."""
    with open(audio_path, "rb") as audio_file:
        transcription = openai.audio.transcriptions.create(
            model=AUDIO_MODEL,
            file=audio_file,
            response_format="text"
        )
    return transcription


def generate_meeting_minutes(transcription):
    """Generates meeting minutes from the transcript using Llama 3.1."""
    system_message = (
        "You are an assistant that produces minutes of meetings from transcripts, "
        "with summary, key discussion points, takeaways, and action items with owners, in markdown."
    )
    user_prompt = (
        f"Below is an extract transcript of a meeting. "
        f"Please write minutes in markdown, including a summary with attendees, location, and date; "
        f"discussion points; takeaways; and action items with owners.\n{transcription}"
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    # Quantization config for CPU usage
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cpu")
    streamer = TextStreamer(tokenizer)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="cpu",
                                                 quantization_config=quant_config)

    # model_no_quantization = AutoModelForCausalLM.from_pretrained(LLAMA, device_map="cpu")
    outputs = model.generate(inputs,max_new_tokens=2000,streamer=streamer)
    response = tokenizer.decode(outputs[0])
    return response


def main():
    """Main function to transcribe audio and generate meeting minutes."""
    print("Transcribing audio...")
    transcription = transcribe_audio(AUDIO_FILE_PATH)
    print("Transcription complete. Generating meeting minutes...")
    minutes = generate_meeting_minutes(transcription)
    print("Meeting minutes generated successfully.")

    # Save minutes to a file
    with open("meeting_minutes.md", "w") as f:
        f.write(minutes)
    print("Meeting minutes saved to meeting_minutes.md")


if __name__ == "__main__":
    main()
