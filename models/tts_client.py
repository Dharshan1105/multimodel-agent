"""
Text-to-Speech client using Kokoro for natural sounding speech.

This is Phase 2 of the TTS implementation. pyttsx3 has been replaced with
Kokoro via the kokoro-onnx library which produces significantly more natural
sounding speech compared to the espeak engine used by pyttsx3.

Kokoro runs fully locally with no API calls. The interface is identical to
the pyttsx3 version so no other files need to change.
"""

import re
import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro


class TTSClient:
    """
    Text-to-Speech client that generates and plays audio using Kokoro.

    Kokoro produces natural sounding speech from text input. Audio is played
    directly through the system speakers and the method blocks until playback
    is fully complete before returning, so the chat loop waits naturally.
    """

    def __init__(self):
        """
        Initialize the Kokoro TTS client.

        Downloads and loads the Kokoro model on first run. Subsequent runs
        load from cache and are much faster.

        Args:
            None

        Returns:
            None
        """
        # Load the Kokoro model
        # Voice options: af_heart, af_bella, af_sarah, am_adam, am_michael
        # af = American Female, am = American Male
        self.kokoro = Kokoro("kokoro-v0_19.onnx", "voices.bin")
        self.voice = "af"
        self.speed = 1.0

    def speak(self, text: str) -> None:
        """
        Generate speech from text and play it, waiting for completion.

        Cleans markdown and emojis from the text first, then generates
        audio using Kokoro and plays it through the system speakers.
        Blocks until playback is fully finished before returning.

        Args:
            text: The text to convert to speech (may contain markdown)

        Returns:
            None (blocks until audio playback is complete)
        """
        # Clean the text before speaking
        clean_text = self._clean_text_for_speech(text)

        if not clean_text.strip():
            return

        # Generate audio samples using Kokoro
        samples, sample_rate = self.kokoro.create(
            clean_text,
            voice=self.voice,
            speed=self.speed,
            lang="en-us"
        )

        # Play the audio and wait until it finishes
        sd.play(samples, sample_rate)
        sd.wait()

    def set_voice(self, voice: str) -> None:
        """
        Change the voice used for speech generation.

        Available voices:
            af_heart   - American female, warm tone (default)
            af_bella   - American female, clear tone
            af_sarah   - American female, soft tone
            am_adam    - American male, deep tone
            am_michael - American male, neutral tone

        Args:
            voice: Voice identifier string from the list above

        Returns:
            None
        """
        self.voice = voice

    def set_speed(self, speed: float) -> None:
        """
        Change the speaking speed.

        Args:
            speed: Speaking speed multiplier.
                   1.0 = normal speed
                   0.8 = slower and clearer
                   1.2 = faster

        Returns:
            None
        """
        self.speed = speed

    def _clean_text_for_speech(self, text: str) -> str:
        """
        Remove markdown formatting and emojis so Kokoro reads clean text.

        Args:
            text: Raw text from Gemma3 possibly containing markdown and emojis

        Returns:
            Clean plain text suitable for speech
        """
        # Remove emojis and non-ASCII unicode symbols
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # Remove bold and italic markers (** and *)
        text = re.sub(r'\*+', '', text)

        # Remove heading markers (# ## ###)
        text = re.sub(r'#+\s*', '', text)

        # Remove backticks for inline code and code blocks
        text = re.sub(r'`+', '', text)

        # Remove markdown links and keep only the display text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

        # Remove underscores used for formatting
        text = re.sub(r'_+', '', text)

        # Remove numbered list markers at start of lines
        text = re.sub(r'^\d+\.\s*', '', text, flags=re.MULTILINE)

        # Clean up extra whitespace and blank lines
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text