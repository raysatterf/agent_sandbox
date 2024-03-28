from openai import OpenAI
from pathlib import Path
from playsound import playsound
import json
import sys
import os

client_openai = OpenAI()
api_key = os.environ.get("OPENAI_API_KEY", None)
working_dir = "C:/dev/proj/demo_crewai-test-2/working_dir"
audio_output_filename = "output_audio.mp3"
audio_output_fullpath = os.path.join(working_dir, audio_output_filename)

# tts_model = "tts-1"
# tts_voice = "nova"
# tts_speed = 0.85

def do_tts(
    tts_text,
    tts_model,
    tts_voice,
    tts_speed,
    # client_openai,
    # working_dir,
    # audio_output_filename="output_audio.mp3",
):
    """
    Generates audio from the given text using OpenAI's text-to-speech,
    plays the audio using playsound, and then removes the temporary file.

    Args:
        text_to_speak (str): The text to be converted to audio.
        client_openai (OpenAI): An OpenAI API client instance.
        working_dir (str): The directory to save the temporary audio file.
        audio_output_filename (str, optional): The name of the audio file. Defaults to "output_audio.mp3".
    """

    response_voice = client_openai.audio.speech.create(
        input=tts_text,
        model=tts_model, # The latest text to speech model, optimized for speed.
        voice=tts_voice,
        speed=tts_speed,
        # response_format="mp3",  # Save as MP3 for compatibility
    )

    response_voice.stream_to_file(audio_output_fullpath)
    playsound(audio_output_fullpath)
    os.remove(audio_output_fullpath)
