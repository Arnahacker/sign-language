import pyttsx3
import threading

class TextToSpeech:
    def __init__(self, rate=160):
        self.rate = rate

    def speak(self, text):
        """Creates a new thread to speak so the video doesn't freeze"""
        threading.Thread(target=self._run_speech, args=(text,)).start()

    def _run_speech(self, text):
        try:
            # Initialize engine inside the thread to prevent conflicts
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            
            # Optional: Try to set a clearer voice (system dependent)
            voices = engine.getProperty('voices')
            if len(voices) > 0:
                engine.setProperty('voice', voices[0].id) 

            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"🔊 TTS Error: {e}")

# Create the instance here so other files can just import 'tts'
tts = TextToSpeech()