import torch
import os
from transformers import pipeline
from diffusers import DiffusionPipeline
from datasets import load_dataset
import soundfile as sf
from IPython.display import Audio
from dotenv import load_dotenv

# Load environment variables (if needed)
load_dotenv(override=True)

# Retrieve API key
HUGGINGFACE_KEY = os.getenv("HF_TOKEN")
print(f"API Key Loaded: {bool(HUGGINGFACE_KEY)}")

# Toggle models easily by setting to True/False
RUN_SENTIMENT_ANALYSIS = False
RUN_NER = False
RUN_QA = True
RUN_SUMMARIZATION = False
RUN_TRANSLATION = False
RUN_CLASSIFICATION = False
RUN_TEXT_GENERATION = False
RUN_IMAGE_GENERATION = False
RUN_AUDIO_GENERATION = False

if RUN_SENTIMENT_ANALYSIS:
    print("\nRunning Sentiment Analysis...")
    classifier = pipeline("sentiment-analysis", device=0 if torch.cuda.is_available() else -1)
    result = classifier("I'm super excited to be on the way to LLM mastery!")
    print(result)

if RUN_NER:
    print("\nRunning Named Entity Recognition...")
    ner = pipeline("ner", grouped_entities=True, device=0 if torch.cuda.is_available() else -1)
    result = ner("Barack Obama was the 44th president of the United States.")
    print(result)

if RUN_QA:
    print("\nRunning Question Answering...")
    question_answerer = pipeline("question-answering", device=0 if torch.cuda.is_available() else -1)
    result = question_answerer(question="Who was the 44th president of the United States?", context="Barack Obama was the 44th president of the United States.")
    print(result)

if RUN_SUMMARIZATION:
    print("\nRunning Text Summarization...")
    summarizer = pipeline("summarization", device=0 if torch.cuda.is_available() else -1)
    text = ("""The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP). "
            "It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering, among others. "
            "It's an extremely popular library that's widely used by the open-source data science community. "
            "It lowers the barrier to entry into the field by providing Data Scientists with a productive, convenient way to work with transformer models.""")
    summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
    print(summary[0]['summary_text'])

if RUN_TRANSLATION:
    print("\nRunning Translation...")
    translator = pipeline("translation", "thilina/mt5-sinhalese-english", device=0 if torch.cuda.is_available() else -1)
    result = translator("The data engineer wanted to code using Python")
    print(result[0]['translation_text'])

if RUN_CLASSIFICATION:
    print("\nRunning Zero-Shot Classification...")
    classifier = pipeline("zero-shot-classification", device=0 if torch.cuda.is_available() else -1)
    result = classifier("Hugging Face's Transformers library is amazing!", candidate_labels=["technology", "sports", "politics"])
    print(result)

if RUN_TEXT_GENERATION:
    print("\nRunning Text Generation...")
    generator = pipeline("text-generation", device=0 if torch.cuda.is_available() else -1)
    result = generator("If there's one thing I want you to remember about using HuggingFace pipelines, it's")
    print(result[0]['generated_text'])

if RUN_IMAGE_GENERATION:
    print("\nRunning Image Generation...")
    image_gen = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    text = "A class of Data Scientists learning about AI, in the surreal style of Salvador Dali"
    image = image_gen(prompt=text).images[0]
    image.show()

if RUN_AUDIO_GENERATION:
    print("\nRunning Audio Generation...")
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts", device='cuda')
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    speech = synthesiser("Hi to an artificial intelligence engineer, on the way to mastery!", forward_params={"speaker_embeddings": speaker_embedding})
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    Audio("speech.wav")