import json
from transformers import pipeline

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize(transcript_path):
    text = json.loads(transcript_path.read_text())["text"]
    return summarizer(text[:3500], max_length=150, min_length=60, do_sample=False)[0]["summary_text"]
