import os
import asyncio
import backoff
import requests
from types import SimpleNamespace
from typing import List

from .base import AsyncBaseAgent

import google.generativeai as genai
from google.generativeai.types import safety_types

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class AsyncGeminiAgent(AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()

    def generate(self, prompt, system_prompt, temperature=None, max_tokens=None):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(self.args.model, system_instruction=system_prompt, safety_settings=safety_settings)
        output = model.generate_content(prompt,
                                        generation_config = genai.GenerationConfig(
                                            max_output_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                            temperature=self.args.temperature if temperature is None else temperature,
                                            )
                                        )
        try:
            text = output.text
        except:
            print("No output from model.")
        return output

    def chat(self, message, temperature=None, max_tokens=None, system_prompt=None, history=None):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(self.args.model,
                                      system_instruction=system_prompt,
                                      safety_settings=safety_settings,
                                      generation_config = genai.GenerationConfig(
                                            max_output_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                            temperature=self.args.temperature if temperature is None else temperature,
                                            )
                                      )
        chat_session = model.start_chat(history=message[:-1])
        response = chat_session.send_message(message[-1])

        return response

    def interact(self, prompt, temperature=None, max_tokens=None, system_prompt=None, history=None):
        if history is None:
            message = self.preprocess_input(prompt)
            output = self.generate(message, system_prompt, temperature=temperature, max_tokens=max_tokens)
            response = self.postprocess_output(output)
        else:
            message = self.preprocess_chat(prompt, history=history)
            output = self.chat(message, system_prompt=system_prompt, history=history, temperature=temperature, max_tokens=max_tokens)
            response = self.postprocess_output(output)
        return response

    def batch_interact(self, prompts, temperature=0, max_tokens=256, system_prompts = None, histories: List[List] = None):
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        elif isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        elif isinstance(system_prompts, list):
            assert len(prompts) == len(system_prompts)

        if histories is not None:
            assert len(prompts) == len(histories)
            message_batch = [self.preprocess_input(prompt, history=history) for prompt, history in zip(prompts, histories)]
        else:
            message_batch = [self.preprocess_input(prompt) for prompt in prompts]
        outputs = asyncio.run(self.batch_generate(message_batch, system_prompts, temperature=None, max_tokens=None))
        responses = [self.postprocess_output(output) for output in outputs]
        return responses

    @backoff.on_exception(backoff.expo, (requests.exceptions.HTTPError))
    async def batch_generate(self, prompts, system_prompts, temperature=None, max_tokens=None):
        loop = asyncio.get_running_loop()
        completions = await asyncio.gather(*[
            loop.run_in_executor(self.executor, self.generate, prompt, system_prompt, temperature, max_tokens)
            for prompt, system_prompt in zip(prompts, system_prompts)
        ])
        return completions

    def preprocess_input(self, text,  system_prompt=None, history=None):
        return text

    def postprocess_output(self, output):
        response = output.text
        return response

    def preprocess_chat(self, text, system_prompt=None, history=None):
        messages = []
        if history is not None:
            for idx, msg in enumerate(history):
                if idx % 2 == 0:
                    messages.append({"role": "user", "parts": f"{msg}"})
                else:
                    messages.append({"role": "model", "parts": f"{msg}"})
        messages.append({"role": "user", "parts": f"{text}"})
        return messages
