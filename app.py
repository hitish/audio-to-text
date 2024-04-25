import streamlit as st
import soundfile as sf
import whisper
from io import BytesIO
import numpy as np

# Set up Whisper model
model = whisper.load_model("base")  # Choose from "base", "medium", or "large"

# Function to transcribe audio
def transcribe(audio_bytes):
  #audio_buffer = BytesIO(audio_bytes)
  audio_array, sample_rate = bytesio_to_numpy(audio_bytes)
  result = model.transcribe(audio_array, task="transcribe")
  return result["text"]

def bytesio_to_numpy(audio_bytes):
  # Load audio data from BytesIO
  with BytesIO(audio_bytes) as audio_buffer:
    audio_data, sample_rate = sf.read(audio_buffer)
  
  # Convert to NumPy array
  audio_array = np.asarray(audio_data)
  audio_array = audio_array.astype(np.float32)
  return audio_array, sample_rate

# Streamlit App
st.title("Speech to Text with Whisper")
st.write("Upload audio File to 'Transcribe'")

audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

if audio_file:
  audio_bytes = audio_file.read()
  transcript = transcribe(audio_bytes)
  st.write("Transcription:", transcript)


st.write("Note: This is a basic example. using base model. For accuracy we need to use large whisper model, which needs high computational resources")
