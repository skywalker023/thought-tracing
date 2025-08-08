from vllm import LLM, SamplingParams
from typing import List

class VllmAgent():
    def __init__(self, model_name, num_gpus=2, max_tokens=1024, **kwargs):
        self.model_name = model_name
        self.model = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95)
        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = 1
        self.cot_prompt = "\nLet's think step by step."
        
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
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        return prompt

    def postprocess_output(self, output):
        return output.outputs[0].text.strip()

    def interact(self, prompt, temperature=1, max_tokens=None, system_prompt: str = None, history: str = None):
        return self.batch_interact([prompt], temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt, histories=[history])[0]

    def batch_interact(self, prompts, temperature=1, max_tokens=None, system_prompt: str = None, histories: List[List] = None):
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature

        if histories is not None:
            assert len(prompts) == len(histories)
            message_batch = [self.preprocess_input(prompt, system_prompt, history) for prompt, history in zip(prompts, histories)]
        else:
            message_batch = [self.preprocess_input(prompt, system_prompt) for prompt in prompts]

        sampling_params = SamplingParams(temperature=temperature, top_p=1, max_tokens=max_tokens)
        # prompts = [self.preprocess_input(text) for text in texts]
        outputs = self.model.generate(message_batch, sampling_params=sampling_params)
        responses = [self.postprocess_output(output) for output in outputs]

        return responses

    def batch_cot(self, prompts, temperature=None, max_tokens=None):
        cot_prompts = [prompt.removesuffix("\nAnswer:") + self.cot_prompt for prompt in prompts]
        cot_responses = self.batch_interact(cot_prompts, temperature, max_tokens)
        return cot_responses

    # def cot(self, prompt, temperature=None, max_tokens=None):
    #     q_prompt = prompt.split("\nAnswer:")[0].strip()
    #     cot_prompt = f"{q_prompt}\nLet's think step by step before answering the question above."
    #     cot_response = self.interact(cot_prompt, temperature, max_tokens)
    #     prompt_with_cot = f"{q_prompt}\n{cot_response}\nTherefore, the answer is:"
    #     final_response = self.interact(prompt_with_cot, temperature, max_tokens)
    #     return final_response

    # def batch_cot(self, prompts, temperature=None, max_tokens=None):
    #     cot_prompts = [prompt.split("\nAnswer:")[0].strip() + "\nLet's think step by step before answering the question above." for prompt in prompts]
    #     cot_responses = self.batch_interact(cot_prompts, temperature, max_tokens)
    #     prompts_with_cot = [prompt.split("\nAnswer:")[0].strip() + f"\n{cot_response}\nTherefore, the answer is:" for prompt, cot_response in zip(prompts, cot_responses)]
    #     return self.batch_interact(prompts_with_cot, temperature, max_tokens)

class NemoAgent(VllmAgent):
    def __init__(self, model_name, num_gpus=4, max_tokens=1024, **kwargs):
        self.model_name = model_name
        self.model = LLM(model=model_name, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.95, max_model_len=416810)
        self.tokenizer = self.model.get_tokenizer()
        self.max_tokens = max_tokens
        self.temperature = 0
        self.cot_prompt = "\nLet's think step by step."

    def interact(self, prompt, temperature=0, max_tokens=None, system_prompt=None, history=None):
        return super().interact(prompt, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt, history=history)

    def batch_interact(self, prompts, temperature=0, max_tokens=256, system_prompt: str = None, histories: List[List] = None):
        return super().batch_interact(prompts, temperature=temperature, max_tokens=max_tokens, system_prompt=system_prompt, histories=histories)
    