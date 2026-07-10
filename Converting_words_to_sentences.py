import requests
from converting_sentences_to_speech import tts


class OllamaHandler:
    def __init__(self, model="llama3"):
        self.model   = model
        self.api_url = "http://localhost:11434/api/chat"

        self.system_prompt = {
            "role": "system",
            "content": (
                "You are a Sign Language interpreter. "
                "I will provide raw glosses (keywords). "
                "Translate them into a spoken English sentence. "
                "Do not explain, just output the translation."
            )
        }
        self.history = [self.system_prompt]

    def process_words(self, words):
        """
        Input:  ['Hello', 'Name', 'Anoop']
        Output: 'Hello, my name is Anoop.'
        """
        if not words:
            return ""

        raw_input = " ".join(words)
        user_msg  = {"role": "user", "content": f"Translate: {raw_input}"}

        # Keep last 10 messages for context window
        current_context = [self.history[0]] + self.history[-10:] + [user_msg]

        try:
            print(f"Sending to {self.model}...")
            response = requests.post(self.api_url, json={
                "model":    self.model,
                "messages": current_context,
                "stream":   False
            })

            if response.status_code == 200:
                result = response.json()["message"]["content"]

                self.history.append(user_msg)
                self.history.append({"role": "assistant", "content": result})

                print(f"AI: {result}")
                tts.speak(result)

                return result
            else:
                print(f"API Error: {response.status_code}")
                return "AI Error"

        except Exception as e:
            print(f"Connection Error: {e}")
            return "Connection Failed"
