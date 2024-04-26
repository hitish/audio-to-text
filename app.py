import streamlit as st
import whisper
import numpy as np

# Set up Whisper model
model = whisper.load_model("base")  # Choose from "base", "medium", or "large"


# Streamlit App
st.title("Speech to Text with Whisper")
st.write("Upload audio File to 'Transcribe'")

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3","weba"])
print(audio_file)
if audio_file:
  audio = whisper.load_audio(audio_file)
    
  transcript = model.transcribe(audio)
  print(transcript)
  st.write("Transcription:", transcript["text"])


st.write("Note: This is a basic example. using base model. For accuracy we need to use large whisper model, which needs high computational resources")
