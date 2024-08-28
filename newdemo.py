import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
import streamlit as st
import tempfile

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform


def transcribe_audio(model_name, non_english, energy_threshold, record_timeout, phrase_timeout, default_microphone):
    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False

    if 'linux' in platform:
        mic_name = default_microphone
        if not mic_name or mic_name == 'list':
            st.write("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                st.write(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    audio_model = whisper.load_model(model_name)

    transcription_placeholder = st.empty()  # Create a placeholder for the transcription

    with source:
        recorder.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    st.write("MODEL LOADED.\n")

    transcription = ""  # Initialize transcription text

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now

                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                result = audio_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language='en')
                text = result['text'].strip()

                # Replace specific phrases in live transcription
                remove = ['Thank You', 'Thank You ', ' Thank you', 'Thank you ', 'Thank you', ' Thank you ', 'thank you', ' Thank you.', 'Thank you. ', '않는','않는 .','않는.']
                for i in remove:
                    text = text.replace(i, "")

                if phrase_complete:
                    transcription = text  # Update transcription for complete phrase
                else:
                    transcription += ' ' + text  # Append ongoing transcription

                # Update the placeholder with the current transcription
                transcription_placeholder.write(transcription)

            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    st.write("\n\nFinal Transcription:")
    st.write(transcription)


def transcribe_uploaded_audio(file, model_name, non_english):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.read())
        temp_file_path = tmp.name

    audio_model = whisper.load_model(model_name)
    result = audio_model.transcribe(temp_file_path, fp16=torch.cuda.is_available(), language='en')

    # The replacement logic is skipped for uploaded audio
    return result['text'].strip()


def main():
    #st.set_page_config(page_title=“TITLE”, page_icon=“./logo.png”, layout=“wide”)
    
    st.image("./logo.png", width =200)
    st.title("Real-Time ASR for Speech Impairment")

    st.sidebar.header("Transcription Options")
    model = st.sidebar.selectbox("Select Whisper model", ["large-v3"])
    non_english = st.sidebar.checkbox("Use non-English model")
    energy_threshold = 1000
    record_timeout = 2.0
    phrase_timeout = 6.0
    default_microphone = st.sidebar.text_input("Default Microphone (for Linux users)", value='pulse')

    # Real-Time Transcription
    if st.sidebar.button("Start Real-Time Transcription"):
        st.subheader("Real-Time Transcription")
        transcribe_audio(model, non_english, energy_threshold, record_timeout, phrase_timeout, default_microphone)

    st.subheader("Upload Audio for Transcription")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        transcription = transcribe_uploaded_audio(uploaded_file, model, non_english)
        st.subheader("Transcription")
        st.write(transcription)


if __name__ == "__main__":
    main()
