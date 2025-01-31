from openai import OpenAI

import os


class ChatBot:
    def __init__(self, key):
        self.model = os.getenv("model")
        self.client = OpenAI(api_key=key)

    def ask(self, text):
        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        """You give short and precise answers."""
                    )
                },
                {"role": "user", "content": text}
            ],
            max_tokens=100,
            temperature=0.7
        )
        return result.choices[0].message.content