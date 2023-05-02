import pickle
import os

import numpy as np

import openai


class ChatBot:
    def __init__(self, api_file = 'api_key.pkl') -> None:
        self.api_file = api_file
        self._setup_api_connection()
        
        self.messages = []
    def _setup_api_connection(self):
        if os.path.exists(self.api_file):
    # Load the API key from the file
            with open(self.api_file, "rb") as f:
                api_key = pickle.load(f)
                print(f"API key loaded from {self.api_file}")
        else:
            # Your API key
            api_key = input("Please enter your API key: ")

            # Save the API key to a file
            with open("api_key.pkl", "wb") as f:
                pickle.dump(api_key, f)
                print("API key saved to api_key.pkl")

        # Use the API key
        openai.api_key = api_key
    def _track_tokens(self):
        pass
    
    def send_msg(self, content):
        msg = {"role":"user", "content":content}
        self.messages.append(msg)
    def get_response(self):
        response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages= self.messages)
        self.messages.append({"role":"assistant","content":response.choices[0]['message'].content})
        print(response.choices[0]['message'].content)
    def chat(self):
        content = None
        while content != 'end':
            content = input("Hello I'm ChatGPT, what do want to ask? ")
            self.send_msg(content)
            try: 
                self.get_response()
            except openai.error.APIConnectionError:
                self._setup_api_connection()




if __name__ =='__main__':
    chatbot = ChatBot()
    chatbot.chat()
