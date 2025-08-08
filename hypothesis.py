import re
import random
import numpy as np
from typing import List
from rich import print
from rich.panel import Panel


class HypothesisV3():
    def __init__(self, target_agent: str, contexts: List[str], perceptions: List[dict], text: str, weight: float, parent_hypothesis: 'HypothesisV3' = None):
        self.target_agent = target_agent
        self.contexts = contexts
        # self.context_history = context_history
        self.perceptions = perceptions
        self.text = text
        self.weight = weight
        # self.observation = observation
        self.parent = parent_hypothesis

    def update_context_history(self, new_context_history):
        self.context_history = new_context_history

    def update_context(self, new_context):
        self.context = new_context

    def update_text(self, new_text):
        self.text = new_text

    def update_weight(self, new_weight):
        self.weight = new_weight

    def add_update_details(self, details):
        self.details = details

    def update_perceptions(self, new_perceptions):
        self.perceptions = new_perceptions

    def __repr__(self) -> str:
        return f"Text: {self.text} Weight: {self.weight}"

class HypothesesSetV3():
    def __init__(self, target_agent: str, contexts: List[dict], perceptions: List[dict], texts: List[str], weights, parent_hypotheses: List[HypothesisV3] = None, previous_ess: float = None, weight_details: dict = None, **kwargs):
        self.target_agent = target_agent
        self.contexts = contexts
        self.perceptions = perceptions
        self.texts = texts
        self.weights = weights
        self.kwargs = kwargs
        # self.observation = observation
        self.previous_ess = previous_ess
        self.weight_details = weight_details
        self.lookahead_scores = kwargs.get('lookahead_scores', None)
        if parent_hypotheses is not None:
            self.hypotheses = [HypothesisV3(target_agent, contexts, perceptions, text, weight, parent_hypothesis=parent) for text, weight, parent in zip(texts, weights, parent_hypotheses)]
        else:
            self.hypotheses = [HypothesisV3(target_agent, contexts, perceptions, text, weight, parent_hypothesis=None) for text, weight in zip(texts, weights)]

    def update_ess(self, ess):
        self.previous_ess = ess

    def update_context_history(self, new_context_history):
        self.context_history = new_context_history
        for hypothesis in self.hypotheses:
            hypothesis.update_context_history(new_context_history)

    def update_context(self, new_context):
        self.context = new_context
        for hypothesis in self.hypotheses:
            hypothesis.context = new_context

    def update_texts(self, new_texts):
        self.texts = new_texts
        for hypothesis, text in zip(self.hypotheses, new_texts):
            hypothesis.update_text(text)

    def update_weights(self, new_weights):
        self.weights = new_weights
        for hypothesis, weight in zip(self.hypotheses, new_weights):
            hypothesis.update_weight(weight)

    def update_perceptions(self, new_perceptions):
        self.perceptions = new_perceptions
        for hypothesis in self.hypotheses:
            hypothesis.update_perceptions(new_perceptions)

    def dump(self):
        """
        Dump the hypotheses set to a dictionary.
        """
        hypotheses_details = []
        for hypothesis in self.hypotheses:
            if 'details' in hypothesis.__dict__:
                details = hypothesis.details
            else:
                details = None
            hypotheses_details.append({
                'details': details
            })
        return {
            'target_agent': self.target_agent,
            'contexts': self.contexts,
            'perceptions': self.perceptions,
            'texts': self.texts,
            'weights': self.weights.tolist(),
            'kwargs': self.kwargs,
            'previous_ess': self.previous_ess,
            'details': hypotheses_details
        }

    def __iter__(self):
        return iter(self.hypotheses)

    def __getitem__(self, idx: int):
        return self.hypotheses[idx]

    

def resample_hypotheses(hypotheses: HypothesesSetV3, ess: float) -> HypothesesSetV3:
    """
    Resample the hypotheses based on the weights.
    """
    resampled_idxs = random.choices(range(len(hypotheses.texts)), hypotheses.weights, k=len(hypotheses.texts))
    target_agent = hypotheses.target_agent
    context_history = hypotheses.context_history
    perceptions = hypotheses.perceptions
    texts = [hypotheses.texts[idx] for idx in resampled_idxs]
    weights = np.ones(len(texts)) / len(texts)
    parents = [hypotheses.hypotheses[idx].parent for idx in resampled_idxs]

    resampled_hypotheses = HypothesesSetV3(target_agent, hypotheses.context, context_history, perceptions, texts, weights, parent_hypotheses=parents, previous_ess=ess)

    return resampled_hypotheses

def resample_hypotheses_with_other_info(hypotheses: HypothesesSetV3, ess: float) -> HypothesesSetV3:
    """
    Resample the hypotheses based on the weights.
    """
    print("Resampling hypotheses @x@x@x@x@x@x")
    for h, w in zip(hypotheses.texts, hypotheses.weights):
        print(Panel(h, title=f"Weight: {w:.2f}", style="purple"))

    resampled_idxs = random.choices(range(len(hypotheses.texts)), hypotheses.weights, k=len(hypotheses.texts))
    target_agent = hypotheses.target_agent
    texts = [hypotheses.texts[idx] for idx in resampled_idxs]
    weights = np.ones(len(texts)) / len(texts)
    parents = [hypotheses.hypotheses[idx].parent for idx in resampled_idxs]
    weight_detail_prompts = [hypotheses.weight_details['prompts'][idx] for idx in resampled_idxs]
    weight_detail_predictions = [hypotheses.weight_details['reasonings'][idx] for idx in resampled_idxs]
    weight_details = {'prompts': weight_detail_prompts, 'reasonings': weight_detail_predictions} #, 'evaluations': weight_detail_evaluations}

    resampled_hypotheses = HypothesesSetV3(target_agent, hypotheses.contexts, hypotheses.perceptions, texts, weights, parent_hypotheses=parents, previous_ess=ess, weight_details=weight_details)

    return resampled_hypotheses

def compute_ess(hypotheses: HypothesesSetV3) -> float:
    """
    Compute the effective sample size of the hypotheses.
    """
    ess = 1 / np.sum(np.square(hypotheses.weights))
    return ess

def backtrack(hypothesis: HypothesisV3) -> List[str]:
    """
    Get the trace of the hypotheses. Following the parent hypothesis until the root hypothesis.
    """
    trace = {'contexts': [], 'perceptions': [], 'texts': [], 'weights': []}
    while hypothesis is not None:
        trace['contexts'].append(hypothesis.context)
        trace['perceptions'].append(hypothesis.perceptions['summary'])
        trace['texts'].append(hypothesis.text)
        trace['weights'].append(hypothesis.weight)
        hypothesis = hypothesis.parent

    # Reverse the trace to get the correct order.
    for key in trace:
        trace[key].reverse()

    # compute the likelihood of the trace.
    trace['likelihood'] = np.prod(trace['weights'])

    return trace

def extract_question(text: str) -> str:
    """
    Extract the question from the text.
    """
    # Extract the question from the text. Extract the line that starts with "Question:" using regex.
    match = re.search(r'Question:(.*)', text)
    if match:
        question = match.group(1).strip()
        rest = text.split(question)[1].strip().split("Answer:")[0].strip().replace("\n", " ")
        question = question.replace("Answer yes or no.", "").strip()
        if rest != "":
            question = f"{question} {rest}"
        return question
    else:
        return "none"

def extract_info_of_question(text: str) -> str:
    """
    Extract the extra info for question in fantom
    """
    # Extract the question from the text. Extract the line that starts with "Question:" using regex.
    if "Target:" in text:
        match = re.search(r'Target:(.*)', text)
    elif "Information:" in text:
        match = re.search(r'Information:(.*)', text)
    else:
        return ""

    if match:
        info = match.group(1).strip()
        return info
    else:
        return ""
