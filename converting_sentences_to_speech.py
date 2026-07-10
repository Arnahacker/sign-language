import pyttsx3
import threading


class TextToSpeech:
    def __init__(self, rate=160):
        self.rate = rate

    def speak(self, text):
        """Runs speech in a daemon thread so the main process can exit cleanly."""
        threading.Thread(target=self._run_speech, args=(text,), daemon=True).start()

    def _run_speech(self, text):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)

            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)

            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")


tts = TextToSpeech()
