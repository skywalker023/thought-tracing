from .gpt import AsyncConversationalGPTBaseAgent, ConversationalGPTBaseAgent, O1BaseAgent, O1MiniAgent
from .gemini import AsyncGeminiAgent
from .together_ai import AsyncTogetherAIAgent, AsyncLlama3Agent, AsyncQwenAgent, AsyncDeepSeekAgent
from .vllm import VllmAgent

def load_model(model_name, num_gpus=2, mode="async", **kwargs):
    if model_name in ["o1-preview-2024-09-12", "o1-2024-12-17", "o3-mini-2025-01-31"]:
        model = O1BaseAgent({'model': model_name, **kwargs})
    elif model_name in ["o1-mini-2024-09-12"]:
        model = O1MiniAgent({'model': model_name, **kwargs})
    elif model_name in ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13", "gpt-4-turbo-2024-04-09", "gpt-3.5-turbo-0301", "gpt-4-0314", "gpt-4-0125-preview", "gpt-4-0613", "gpt-3.5-turbo-0125"]:
        if mode == "async":
            model = AsyncConversationalGPTBaseAgent({'model': model_name, **kwargs})
        elif mode == "non-async":
            model = ConversationalGPTBaseAgent({'model': model_name, **kwargs})
    elif model_name in ["gpt-4-turbo-nonasync", "gpt-4o-nonasync"]:
        model = ConversationalGPTBaseAgent({'model': model_name.removesuffix("-nonasync"), **kwargs})
    elif model_name.startswith("gemini-"):
        model = AsyncGeminiAgent({'model': model_name, **kwargs})
    elif model_name in ["meta-llama/Llama-3-70b-chat-hf-tg", "meta-llama/Llama-3-8b-chat-hf-tg", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo"]:
        model = AsyncLlama3LogProbAgent({'model': model_name, 'temperature': 0, 'max_tokens': 1024, **kwargs})
    elif model_name in ["meta-llama/Llama-2-13b-chat-hf", "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Meta-Llama-3-8B-Instruct"]:
        model = VllmAgent(model_name, num_gpus=num_gpus, **kwargs)
    elif model_name in ["Qwen/Qwen2.5-72B-Instruct-Turbo", "Qwen/QwQ-32B-Preview"]:
        model = AsyncQwenAgent({'model': model_name, 'temperature': 0, 'max_tokens': 1024, **kwargs})
    elif model_name in ["deepseek-ai/DeepSeek-R1"]:
        model = AsyncDeepSeekAgent({'model': model_name, 'temperature': 0, 'max_tokens': 5000, **kwargs})
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return model
