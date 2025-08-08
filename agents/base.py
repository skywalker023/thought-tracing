import time
import asyncio
import backoff
import requests
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

class BaseAgent(ABC):
    def __init__(self):
        self.cot_prompt = "\nLet's think step by step."
    
    @abstractmethod
    def generate(self, prompt):
        pass
    
    @abstractmethod
    def batch_interact(self, prompts):
        pass

    @abstractmethod
    def interact(self, prompt):
        pass

    @abstractmethod
    def preprocess_input(self, text):
        pass

    @abstractmethod
    def postprocess_output(self, output):
        pass

    def cot(self, prompt, temperature=None, max_tokens=None):
        cot_prompt = prompt + self.cot_prompt
        cot_response = self.interact(cot_prompt, temperature, max_tokens)
        return cot_response

class AsyncBaseAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=16)

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

    @abstractmethod
    def generate(self, prompt, temperature=None, max_tokens=None):
        pass

    def interact(self, prompt, temperature=None, max_tokens=None):
        prompt = self.preprocess_input(prompt)
        output = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
        response = self.postprocess_output(output)
        return response

    @backoff.on_exception(backoff.expo, (requests.exceptions.HTTPError))
    async def batch_generate(self, prompts, temperature=None, max_tokens=None):
        loop = asyncio.get_running_loop()
        completions = await asyncio.gather(*[
            loop.run_in_executor(self.executor, self.generate, prompt, temperature, max_tokens)
            for prompt in prompts
        ])
        return completions

    def batch_interact(self, prompts, temperature=0, max_tokens=256, system_prompts = None, histories: List[List] = None):
        if system_prompts is None:
            system_prompts = [None] * len(prompts)
        elif isinstance(system_prompts, str):
            system_prompts = [system_prompts] * len(prompts)
        elif isinstance(system_prompts, list):
            assert len(prompts) == len(system_prompts)

        if histories is not None:
            assert len(prompts) == len(histories)
            message_batch = [self.preprocess_input(prompt, system_prompt, history) for prompt, system_prompt, history in zip(prompts, system_prompts, histories)]
        else:
            message_batch = [self.preprocess_input(prompt, system_prompt) for prompt, system_prompt in zip(prompts, system_prompts)]
        outputs = asyncio.run(self.batch_generate(message_batch, temperature=None, max_tokens=None))
        responses = [self.postprocess_output(output) for output in outputs]

        return responses

    def batch_cot(self, prompts, temperature=None, max_tokens=None):
        cot_prompts = [prompt.removesuffix("\nAnswer:") + self.cot_prompt for prompt in prompts]
        cot_responses = self.batch_interact(cot_prompts, temperature, max_tokens)
        return cot_responses
