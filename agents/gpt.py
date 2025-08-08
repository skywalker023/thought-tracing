# https://github.com/openai/openai-python
import os
import time
import json
import asyncio
import openai
import backoff
from openai import OpenAI, AsyncOpenAI
from types import SimpleNamespace
from .base import BaseAgent, AsyncBaseAgent
from typing import List, Tuple

class GPT3BaseAgent(BaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__()
        self.args = SimpleNamespace(**kwargs)
        self._set_default_args()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def _set_default_args(self):
        if not hasattr(self.args, 'model'):
            self.args.model = "gpt-3.5-turbo-instruct"
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 1024
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 1.0
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0
        if not hasattr(self.args, 'n'):
            self.args.n = 1

    def generate(self, prompt, temperature=None, max_tokens=None):
        while True:
            try:
                completion = self.client.completions.create(model=self.args.model,
                                                            prompt=prompt,
                                                            temperature=self.args.temperature if temperature is None else temperature,
                                                            max_tokens=self.args.max_tokens if max_tokens is None else max_tokens,
                                                            top_p=self.args.top_p,
                                                            frequency_penalty=self.args.frequency_penalty,
                                                            presence_penalty=self.args.presence_penalty,
                                                            stop=self.args.stop_tokens if hasattr(self.args, 'stop_tokens') else None,
                                                            logprobs=self.args.logprobs if hasattr(self.args, 'logprobs') else 0,
                                                            echo=self.args.echo if hasattr(self.args, 'echo') else False,
                                                            n=self.args.n if hasattr(self.args, 'n') else 1)
                break
            except (RuntimeError, openai.RateLimitError, openai.APIError, openai.APIConnectionError) as e:
                print("Error: {}".format(e))
                time.sleep(0.2)
                continue

        return completion
    
    def preprocess_input(self, text):
        return text

    def postprocess_output(self, outputs):
        responses = [c.text.strip() for c in outputs.choices]

        return responses[0]

    def parse_ordered_list(self, numbered_items):
        ordered_list = numbered_items.split("\n")
        output = [item.split(".")[-1].strip() for item in ordered_list if item.strip() != ""]

        return output

    def interact(self, prompt, temperature=None, max_tokens=None):
        outputs = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        responses = self.postprocess_output(outputs)

        return responses

class ConversationalGPTBaseAgent(GPT3BaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)

    def _set_default_args(self):
        if not hasattr(self.args, 'temperature'):
            self.args.temperature = 0
        if not hasattr(self.args, 'max_tokens'):
            self.args.max_tokens = 1024
        if not hasattr(self.args, 'top_p'):
            self.args.top_p = 1.0
        if not hasattr(self.args, 'frequency_penalty'):
            self.args.frequency_penalty = 0
        if not hasattr(self.args, 'presence_penalty'):
            self.args.presence_penalty = 0

    def generate(self, prompt, temperature=None, max_tokens=None):
        while True:
            try:
                completion = self.client.chat.completions.create(model=self.args.model,
                                                                 messages=[{"role": "user", "content": f"{prompt}"}],
                                                                 temperature=self.args.temperature if temperature is None else temperature,
                                                                 max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
                break
            except (openai.APIError, openai.RateLimitError) as e:
                print("Error: {}".format(e))
                time.sleep(1)
                continue

        return completion

    def json_generate(self, prompt, temperature=None, max_tokens=None):
        while True:
            try:
                completion = self.client.chat.completions.create(model=self.args.model,
                                                                 response_format={ "type": "json_object" },  
                                                                 messages=[
                                                                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                                                                    {"role": "user", "content": f"{prompt}"}
                                                                    ],
                                                                 temperature=self.args.temperature if temperature is None else temperature,
                                                                 max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
                break
            except (openai.APIError, openai.RateLimitError) as e:
                print("Error: {}".format(e))
                time.sleep(1)
                continue

        return completion

    def preprocess_input(self, text, system_prompt=None, history=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": f"{system_prompt}"})
        if history is not None:
            for idx, msg in enumerate(history):
                if idx % 2 == 0:
                    messages.append({"role": "user", "content": f"{msg}"})
                else:
                    messages.append({"role": "assistant", "content": f"{msg}"})
        messages.append({"role": "user", "content": f"{text}"})
        return messages

    def postprocess_output(self, outputs):
        responses = [c.message.content.strip() for c in outputs.choices]

        return responses[0]

    def interact(self, prompt, temperature=0, max_tokens=1024, system_prompt=None, history=None, json_mode=False):
        if json_mode:
            output = self.json_generate(prompt, temperature=temperature, max_tokens=max_tokens)
        else:
            output = self.generate(prompt, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt, history=history)
        response = self.postprocess_output(output)

        return response

    def generate(self, prompt, temperature=None, max_tokens=None, system_prompt=None, history=None):
        message = self.preprocess_input(prompt, system_prompt=system_prompt, history=history)
        while True:
            try:
                completion = self.client.chat.completions.create(model=self.args.model,
                                                                 messages=message,
                                                                 temperature=self.args.temperature if temperature is None else temperature,
                                                                 max_tokens=self.args.max_tokens if max_tokens is None else max_tokens)
                break
            except (openai.APIError, openai.RateLimitError) as e:
                print("Error: {}".format(e))
                time.sleep(1)
                continue

        return completion

    def batch_interact(self, prompts, temperature=1, max_tokens=256):
        raise NotImplementedError

class AsyncConversationalGPTBaseAgent(ConversationalGPTBaseAgent, AsyncBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)
        self._set_default_args()
        self.client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    def preprocess_input(self, text, system_prompt=None, history=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": f"{system_prompt}"})
        if history is not None:
            for idx, msg in enumerate(history):
                if idx % 2 == 0:
                    messages.append({"role": "user", "content": f"{msg}"})
                else:
                    messages.append({"role": "assistant", "content": f"{msg}"})
        messages.append({"role": "user", "content": f"{text}"})
        return messages

    async def batch_generate(self, message_batch, temperature=0, max_tokens=1024):
        completions = await asyncio.gather(*[self.client.chat.completions.create(model=self.args.model,
                                                                                 messages=messages,
                                                                                 temperature=temperature,
                                                                                 max_tokens=max_tokens)
                                            for messages in message_batch])
        return completions

    def batch_interact(self, prompts, temperature=0, max_tokens=1024, system_prompts = None, histories: List[List] = None):
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        elif isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        elif isinstance(system_prompts, list):
            assert len(prompts) == len(system_prompts)

        if histories is not None and histories[0] is not None:
            assert len(prompts) == len(histories)
            message_batch = [self.preprocess_input(prompt, system_prompt, history) for prompt, system_prompt, history in zip(prompts, system_prompts, histories)]
        else:
            message_batch = [self.preprocess_input(prompt, system_prompt) for prompt, system_prompt in zip(prompts, system_prompts)]
        while True:
            try:
                outputs = asyncio.run(self.batch_generate(message_batch, temperature=temperature, max_tokens=max_tokens))
            except Exception as e:
                print("Error: {}".format(e))
                time.sleep(0.1)
                continue
            break
        responses = [self.postprocess_output(output) for output in outputs]

        return responses

    def interact(self, prompt, temperature=0, max_tokens=1024, system_prompt=None, history=None):
        outputs = self.batch_interact([prompt], temperature=temperature, max_tokens=max_tokens, system_prompts=system_prompt, histories=[history])

        return outputs[0]

    def batch_cot_fauxpas_eai(self, prompts, temperature=None, max_tokens=None):
        cot_prompts = [prompt + "\nLet's think step by step." for prompt in prompts]
        cot_responses = self.batch_interact(cot_prompts, temperature, max_tokens)
        prompts_with_cot = []
        for idx, (prompt, cot_response) in enumerate(zip(prompts, cot_responses)):
            if idx in [0, 3]:
                prompts_with_cot.append(f'{prompt}\n{cot_response}\nAnswer with "Yes" or "No" only, without explanations. In case of doubt, answer according to the most probable answer. Therefore, the answer is:')
            elif idx == 1:
                prompts_with_cot.append(f'{prompt}\n{cot_response}\nAnswer with a quote only without explanations. Therefore, the answer is:')
            elif idx == 2:
                prompts_with_cot.append(f'{prompt}\n{cot_response}\nAnswer the question only, without explanations. Therefore, the answer is:')
        final_responses = self.batch_interact(prompts_with_cot, temperature, max_tokens)
        return final_responses

class O1BaseAgent(AsyncConversationalGPTBaseAgent):
    def __init__(self, kwargs: dict):
        super().__init__(kwargs)
        self.reasoning_effort = kwargs.get('reasoning_effort')

    async def batch_generate(self, message_batch, temperature=0, max_tokens=None):
        completions = await asyncio.gather(*[self.client.chat.completions.create(model=self.args.model,
                                                                                 messages=messages,
                                                                                 reasoning_effort=self.reasoning_effort)
                                            for messages in message_batch])
        return completions

    def preprocess_input(self, text, system_prompt=None, history=None):
        messages = []
        # if system_prompt is not None:
        #     messages.append({"role": "system", "content": f"{system_prompt}"})
        if history is not None:
            for idx, msg in enumerate(history):
                if idx % 2 == 0:
                    if system_prompt is not None:
                        messages.append({"role": "user", "content": f"{system_prompt}\n\n{msg}"})
                    else:
                        messages.append({"role": "user", "content": f"{msg}"})
                else:
                    messages.append({"role": "assistant", "content": f"{msg}"})
            messages.append({"role": "user", "content": f"{text}"})
        else:
            messages.append({"role": "user", "content": f"{system_prompt}\n\n{text}"})
            
        return messages

    def postprocess_output(self, outputs):
        responses = [c.message.content.strip() for c in outputs.choices]

        # return responses[0]
        return {"response": responses[0]}

    def batch_interact(self, prompts, temperature=0, max_tokens=1024, system_prompts=None, histories: List[List] = None, reasoning_effort=None):
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        elif isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        elif isinstance(system_prompts, list):
            assert len(prompts) == len(system_prompts)

        if histories is not None and histories[0] is not None:
            assert len(prompts) == len(histories)
            message_batch = [self.preprocess_input(prompt, system_prompt, history) for prompt, system_prompt, history in zip(prompts, system_prompts, histories)]
        else:
            message_batch = [self.preprocess_input(prompt, system_prompt) for prompt, system_prompt in zip(prompts, system_prompts)]
        while True:
            try:
                outputs = asyncio.run(self.batch_generate(message_batch, temperature=temperature, max_tokens=max_tokens))
            except Exception as e:
                print("Error: {}".format(e))
                time.sleep(0.1)
                continue
            break
        responses = [self.postprocess_output(output) for output in outputs]

        return responses

    def interact(self, prompt, temperature=0, max_tokens=1024, system_prompt=None, history=None):
        outputs = self.batch_interact([prompt], temperature=temperature, max_tokens=max_tokens, system_prompts=system_prompt, histories=[history], reasoning_effort=self.reasoning_effort)

        return outputs[0]

class O1MiniAgent(O1BaseAgent):
    async def batch_generate(self, message_batch, temperature=0, max_tokens=None):
        completions = await asyncio.gather(*[self.client.chat.completions.create(model=self.args.model,
                                                                                 messages=messages
                                                                                )
                                            for messages in message_batch])
        return completions