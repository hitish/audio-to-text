import streamlit as st
import whisper
import tempfile

# Set up Whisper model
model = whisper.load_model("base")  # Choose from "base", "medium", or "large"


# Streamlit App
st.title("Speech to Text with Whisper")
st.write("Upload audio File to 'Transcribe'")

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if audio_file:
  with tempfile.NamedTemporaryFile(delete=False, suffix="."+audio_file.name.split(".")[-1]) as temp_file:
      temp_file.write(audio_file.read())
  audio = whisper.load_audio(temp_file.name)
  transcript = model.transcribe(audio)
  st.write("Transcription:", transcript["text"])


st.write("Note: This is a basic example. using base model. For accuracy we need to use large whisper model, which needs high computational resources")
