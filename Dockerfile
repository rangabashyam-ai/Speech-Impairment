# Base image
FROM python:3.10-slim

# Install necessary packages
RUN apt-get update && apt-get install -y \
    pulseaudio \
    alsa-utils \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for PulseAudio
ENV PULSE_SERVER=unix:/run/pulse/native

# Create a directory for your app
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy your application code
COPY . .

# Expose the port Streamlit will run on
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "mic.py"]
