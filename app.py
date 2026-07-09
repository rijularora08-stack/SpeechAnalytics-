import gradio as gr
import os
import whisper
from transformers import pipeline

# Load AI models
whisper_model = whisper.load_model("base")

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=1
)

def analyze_audio(audio):
    if audio is None:
        return "Please upload an audio file.", "", ""

    # Speech-to-Text
    result = whisper_model.transcribe(audio)
    transcript = result["text"]

    # Summary
if len(transcript.split()) > 50:
    summary = summarizer(
        transcript,
        max_length=100,
        min_length=30,
        do_sample=False
    )[0]["summary_text"]
else:
    summary = transcript
           
# ---------------- Gradio Interface ----------------

demo = gr.Interface(
    fn=analyze_audio,
    inputs=gr.Audio(
        type="filepath",
        label="Upload Audio File"
    ),
    outputs=[
        gr.Textbox(label="Transcript", lines=10),
        gr.Textbox(label="Summary", lines=5),
        gr.Textbox(label="Detected Emotion")
    ],
    title="Speech Analytics Tool",
    description="Upload an audio file to generate transcription, summary, and emotion analysis."
)

if __name__ == "__main__":
    demo.launch(share=True)
