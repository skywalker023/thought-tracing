from copy import deepcopy
import re
import os
import json
import argparse
from typing import List
from abc import ABC, abstractmethod
import numpy as np

from rich import print, box
from rich.panel import Panel
import colorful as cf
cf.use_true_colors()
cf.use_style('monokai')

from agents.load_model import load_model
from utils import (
    load_prompt,
    softmax,
    prompting_for_ordered_list,
    overall_jaccard_similarity,
    list_to_unordered_list_string,
    capture_and_parse_ordered_list,
    NpEncoder
)
from hypothesis import compute_ess, extract_question, resample_hypotheses_with_other_info, HypothesesSetV3

def get_tracer_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-tracing', action='store_true', help='whether to run the model with thought tracing')
    parser.add_argument('--tracing-model', type=str, help='Model to use to answer final question.')
    parser.add_argument('--n-hypotheses', type=int, default=4, help='number of hypotheses to generate for each input', )
    parser.add_argument('--target-perceptions', type=str, default='sight',help='target perceptions to test')  #'sight,hearing,overall', 
    parser.add_argument('--use-helper-llm', action='store_true', help='whether to use user helper llm for identifying target agent and labeling actions',)
    parser.add_argument('--existing-traces', default=None, help='path to existing traces')
    parser.add_argument('--input-is-chat', action='store_true', help="whether the input is a chat or not")
    parser.add_argument('--dataset', type=str, required=True, help='dataset')
    parser.add_argument('--likelihood-estimate', default="prompting", type=str, choices=['rollout', 'prompting'], help='likelihood estimation method')
    parser.add_argument('--tracer-type', type=str, help='tracer type')
    return parser

class BaseTracer(ABC):
    def __init__(self, args):
        self.tracer_model = load_model(args.tracing_model, **args.__dict__)
        self.tracer_model.args.model = self.tracer_model.args.tracing_model
        if args.tracing_model == args.model:
            self.base_model = self.tracer_model
        else:
            self.base_model = load_model(args.model, **args.__dict__)
        tracer_name = args.tracing_model.replace("/", "-")
        base_name = args.model.replace("/", "-")
        self.output_file = os.path.join(args.output_dir, f"tracer-{tracer_name}_model-{base_name}_runid-{args.run_id}_nhypotheses-{args.n_hypotheses}.jsonl")
        self.trace_header = "Let's trace [target agent]'s thoughts step by step through the context.\n"
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)

    def identify_target(self, input_text: str) -> str:
        """
        Identify the target agent that we have to trace by looking at the question.

        Args:
            input_text (str): The context text.

        Returns:
            str: The target agent.
        """
        question = extract_question(input_text)
        target_identification_prompt = f"'{question}'\n\nMain question: Who is the subject of the above question? Whose perspective is this question primarily about? Provide the name of the individual, their title, or the group. If the subject of the question is not related to a person or a group, state 'none'.\nThe concise answer to the main question is (e.g, name):"

        if self.args.use_helper_llm:
            llm = load_model('gpt-4o', run_id=self.args.run_id)
            output = llm.interact(target_identification_prompt, temperature=0, max_tokens=16)
        else:
            output = self.tracer_model.interact(target_identification_prompt, temperature=0, max_tokens=16)
        target_agent = output.split("\n")[0].split(":")[-1].strip().strip(".").replace("*", "")

        return target_agent

    def label_action(self, agent: str, input_text: str) -> List[dict]:
        """
        Segment the text into interleaved chunks regarding the target agent's actions.

        Args:
            agent (str): The target character.
            input_text (str): The context text.

        Returns:
            List[dict]: A list of dictionaries containing the action label and the text.
        """
        action_labeling_prompt = load_prompt(f'label_actions_{self.args.dataset}.txt')
        text = input_text.strip().removesuffix("Answer:").strip()
        match = re.search(r'(.+?)(Output:|Choose one of the following:)', text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        action_labeling_prompt = action_labeling_prompt.replace('<<context>>', text)
        action_labeling_prompt = action_labeling_prompt.replace('<<target_character>>', agent)

        if self.args.use_helper_llm:
            llm = load_model('gpt-4o', run_id=self.args.run_id)
            labeled_text = llm.interact(action_labeling_prompt, temperature=0)
        else:
            labeled_text = self.tracer_model.interact(action_labeling_prompt, temperature=0)

        return labeled_text

    def label_action_for_chat(self, target_agent: str, text: str) -> List[dict]:
        if "\nInformation: " in text:
            separator = "\nInformation: "
        elif "\nTarget: " in text:
            separator = "\nTarget: "
        else:
            separator = "\nQuestion: "
        convo = text.split(separator)[0].split("Meeting:")[-1].strip().split("\n")
        labeled_convo = []
        for line in convo:
            line = line.strip()
            if line != "":
                if line.startswith(target_agent + ":"):
                    line = line.strip() + "<action>"
                else:
                    # capture only the text with alphabets in the line and check whether that text starts with the character's name
                    patterns = re.findall(r'[a-zA-Z]+', line)
                    if patterns[0].startswith(target_agent):
                        line = line.strip() + "<action>"
                    else:
                        line = line.strip() + "<no action>"
                labeled_convo.append(line)

        # merge actions that are split into multiple lines in conversations. This is needed because we don't get perception for utterances from the target speaker.
        for idx, line in enumerate(labeled_convo):
            if line.endswith("<action>"):
                if idx + 1 < len(labeled_convo) and labeled_convo[idx + 1].endswith("<action>"):
                    labeled_convo[idx] = labeled_convo[idx].removesuffix("<action>") + "\n" + labeled_convo[idx + 1].removesuffix("<action>") + "<action>"
                    labeled_convo.pop(idx + 1)

        return "\n".join(labeled_convo)

    def label_action_for_mmtom(self, target_agent: str, text: str) -> List[dict]:
        context, action_text = text.split("\nActions")
        context = context.strip() + "<no action>"
        action_text = "\nActions" + action_text
        _actions = action_text.split(". ")
        actions = []
        for idx, a in enumerate(_actions):
            if idx == 0:
                a = a.strip(".") + ".<no action>" # the first action is not an action -- e.g., David is situated in the kitchen.
                actions.append(a)
            else:
                if a != "":
                    a = a.strip(".") + ".<action>"
                    actions.append(a)

        labeled_text = context + "".join(actions)
        return labeled_text

    def interleave_states_and_actions(self, labeled_text: str, agent: str) -> List[dict]:
        """
        Segment the text into interleaved chunks regarding the target agent's actions and states
        Returns:
            List[dict]: A list of dictionaries containing the action label and the text.
        """

        sentences = labeled_text.split(">")
        sentences = [sentence.strip() for sentence in sentences if sentence.strip() != ""]

        # Group parts with no actions, so that the text is segmented into interleaved chunks of actions and states
        segmented_text = []
        if self.args.input_is_chat:
            separator = "\n"
        else:
            separator = " "
        state = ""
        actions = ""
        for sentence in sentences:
            if sentence.endswith("<no action"):
                no_action_sentence = sentence.removesuffix("<no action").strip()
                if actions != "":
                    segmented_text.append({'action': True, 'text': actions})
                    actions = ""

                if state == "":
                    state = no_action_sentence
                else:
                    state = state + separator + no_action_sentence

            elif sentence.endswith("<action"):
                action_sentence = sentence.removesuffix("<action").strip()
                if state != "":
                    segmented_text.append({'action': False, 'text': state})
                    state = ""

                if actions == "":
                    actions = action_sentence
                else:
                    segmented_text.append({'action': True, 'text': actions})
                    # actions = actions + separator + action_sentence
                    actions = action_sentence
            else:
                clean_sentence = sentence.strip()
                if actions != "":
                    segmented_text.append({'action': True, 'text': actions})
                    actions = ""

                if state == "":
                    state = clean_sentence
                else:
                    state = state + separator + clean_sentence

        if state != "":
            segmented_text.append({'action': False, 'text': state})
        if actions != "":
            segmented_text.append({'action': True, 'text': actions})

        return segmented_text

    def set_trajectory(self, state_action_segments: List[dict]) -> List[dict]:
        """
        Set the trajectory of the target agent.

        Args:
            state_action_segments (List[dict]): A list of dictionaries containing the action label and the text.

        Returns:
            List[dict]: A list of dictionaries containing the action label and the text.
        """
        trajectory = []
        for idx, segment in enumerate(state_action_segments):
            if segment['action']:
                if idx - 1 >= 0:
                    previous = state_action_segments[idx - 1]
                    if previous['action']:
                        trajectory.append({'state': None, 'action': segment['text']})
                    else:
                        trajectory.append({'state': previous['text'], 'action': segment['text']})
                else:
                    trajectory.append({'state': None, 'action': segment['text']})
        if segment['action'] is False:
            trajectory.append({'state': segment['text'], 'action': None})

        if len(trajectory) == 0:
            for idx, segment in enumerate(state_action_segments):
                trajectory.append({'state': segment['text'], 'action': None})

        return trajectory

    def preprocess_input(self, text: str, target_agent=None) -> dict:
        """
        Preprocess the input_text before passing it to the tracer model. Identify the target agent (i.e., character) and label the actions.

        Args:
            text (str): The context text.

        Returns:
            str: The target agent.
            List[dict]: A list of dictionaries containing the action label and the text.
        """
        if target_agent is None:
            target_agent = self.identify_target(text)

        if target_agent.lower() == "none":
            print(Panel(text, title="Input Text: with target character 'none'?", style="red", expand=False, box=box.SIMPLE_HEAD))
            return None

        if self.args.print:
            print(Panel(text, title="Input Text", style="blue", expand=False, box=box.SIMPLE_HEAD))
        context = text.split("\nQuestion:")[0]

        if self.args.input_is_chat:
            action_labeled_text = self.label_action_for_chat(target_agent, context)
        elif self.args.dataset == "mmtom":
            action_labeled_text = self.label_action_for_mmtom(target_agent, context)
        else:
            action_labeled_text = self.label_action(target_agent, context)

        state_action_segments = self.interleave_states_and_actions(action_labeled_text, target_agent)
        trajectory = self.set_trajectory(state_action_segments)
        if len(trajectory) == 0:
            print(Panel(text, title="Input Text: No actions found", style="red", expand=False, box=box.SIMPLE_HEAD))

        if self.args.print:
            print(cf.bold | cf.green("Target character: " + target_agent))
            print()

        question = extract_question(text)

        return {'question': question, 'action_labeled_text': state_action_segments, 'trajectory': trajectory, 'context': context, 'target_agent': target_agent}

    @abstractmethod
    def track_perception(self, target_agent: str, context: str) -> dict:
        """
        Track the perceptions of the target character or agent.

        Args:
            target_agent (str): The target character.
            context (str): The context text.

        Returns:
            dict: A dictionary containing the perception of the target agent and the prompt used for inference.
            {'text': perception_inference, 'prompt': perception_query}
        """
        pass

    @abstractmethod
    def initialize(self, target_agent, context):
        pass

    @abstractmethod
    def trace(self, text):
        pass

    def batch_trace(self, texts, use_tracings=None):
        if use_tracings is None:
            responses = [self.trace(text) for text in texts]
        else:
            responses = []
            for use_tracing, text in zip(use_tracings, texts):
                if use_tracing:
                    responses.append(self.trace(text))
                else:
                    responses.append("")
        return responses

    def dump(self, traced_thought: dict, hypotheses_list: List[HypothesesSetV3]):
        """
        Dump to jsonl

        Args:
            traced_thought (dict): _description_
            hypotheses_list (List[HypothesesSet]): _description_
        """
        dumped_hypotheses_list = [h.dump() for h in hypotheses_list]
        traced_thought['hypotheses'] = dumped_hypotheses_list
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(traced_thought, cls=NpEncoder) + '\n')

    def interact(self, text: str, temperature=0, max_tokens: int=256):
        return self.base_model.interact(text, temperature=temperature)

    def batch_interact(self, texts: list, temperature: float=0, max_tokens: int=256):
        return self.base_model.batch_interact(texts, temperature=temperature, max_tokens=max_tokens)

    def batch_cot(self, texts: list, temperature: float=0, max_tokens: int=256):
        return self.base_model.batch_cot(texts, temperature=temperature, max_tokens=max_tokens)

class Tracer(BaseTracer):
    def __init__(self, args):
        super().__init__(args)
        self.trace_base_header = "To answer this question, let's analyze the context step by step regarding [target agent]'s perceptions and thoughts."
        self.cache_db = {}

    def set_tracer_variables(self, preprocessed_text):
        self.question = preprocessed_text['question']
        self.input_context = preprocessed_text['context']
        self.target_agent = preprocessed_text['target_agent']
        self.trace_header = self.trace_base_header.replace("[target agent]", self.target_agent)

    def get_perception_tracking_prompts(self, state_action: dict, context_history: List[dict] = None, target_agent: str = None) -> List[str]:
        target_agent = self.target_agent if target_agent is None else target_agent
        state = state_action['state']
        action = state_action['action']
        context_history_list = [c['text'] for c in context_history] if context_history is not None else []
        prompts = []
        sysprompts = []

        if state:
            sysprompt_for_state = f"You are an expert perception tracker tasked with determining whether {target_agent} perceived the target context. Briefly describe what {target_agent} saw or why {target_agent} could not see the target context. Make your response concise."
            if len(context_history) > 0:
                context_input_for_state = ""
                context_input_for_state += list_to_unordered_list_string(context_history_list, list_bullet="")
                context_input_for_state += f"\n<target context>\n{state}\n</target context>"
            else:
                context_input_for_state = f"<context>\n{state}\n</context>"
            query_for_state = f"{context_input_for_state}" 
            prompts.append(query_for_state)
            sysprompts.append(sysprompt_for_state)

        if action:
            sysprompt_for_action = f"You are an expert perception tracker tasked with determining what {target_agent} have perceived during {target_agent}'s action/utterance. Briefly describe what {target_agent} saw during the new actions/utterance. Make your response concise."
            context_input_for_action = "<context>\n"
            if state:
                context_history_list.append(state)
            context_input_for_action += list_to_unordered_list_string(context_history_list, list_bullet="")
            context_input_for_action += f"\n</context>"
            context_input_for_action += f"\n\n<action>{action}</action>"
            query_for_action = f"{context_input_for_action}"
            prompts.append(query_for_action)
            sysprompts.append(sysprompt_for_action)
        return {'prompts': prompts, 'sysprompts': sysprompts}

    def get_perception_tracking_prompts_for_chat(self, state_action: dict, context_history: List[dict] = None, target_agent: str = None) -> List[str]:
        target_agent = self.target_agent if target_agent is None else target_agent
        state = state_action['state']
        action = state_action['action']
        context_history_list = [c['text'] for c in context_history] if context_history is not None else []
        prompts = []
        sysprompts = []

        if state:
            sysprompt_for_state = f"You are an expert perception tracker tasked with determining whether {target_agent} was involved in the conversation or was not. If the {target_agent} was in the scene, they must have perceived the context. If they were away, they did not perceive the context."
            if len(context_history) > 0:
                context_input_for_state = ""
                context_input_for_state += list_to_unordered_list_string(context_history_list, list_bullet="")
                context_input_for_state += f"\n<target context>\n{state}\n</target context>"
            else:
                context_input_for_state = f"<context>\n{state}\n</context>"
            if action:
                context_input_for_state += f"\n<response>\n{action}\n</response>"
            query_for_state = f"{context_input_for_state}" 
            prompts.append(query_for_state)
            sysprompts.append(sysprompt_for_state)

        return {'prompts': prompts, 'sysprompts': sysprompts}

    def track_perception(self, trajectory, target_agent=None):
        prompts = []
        sysprompts = []
        context_history = []
        if target_agent is None:
            target_agent = self.target_agent

        for idx, t in enumerate(trajectory):
            if self.args.input_is_chat:
                perception_prompts = self.get_perception_tracking_prompts_for_chat(t, context_history, target_agent)
            else:
                perception_prompts = self.get_perception_tracking_prompts(t, context_history, target_agent)
            prompts.extend(perception_prompts['prompts'])
            sysprompts.extend(perception_prompts['sysprompts'])
            if t['state']:
                context_history.append({'text': t['state'], 'action': False})
            if t['action']:
                context_history.append({'text': t['action'], 'action': True})

        perception_inferences = self.tracer_model.batch_interact(prompts, temperature=0, system_prompts=sysprompts)
        perception_trajectory = [{'state': None, 'action': None} for _ in range(len(trajectory))]
        for idx, c in enumerate(trajectory):
            if c['state']:
                perception_trajectory[idx]['state'] = perception_inferences.pop(0)
            if c['action'] and not self.args.input_is_chat:
                perception_trajectory[idx]['action'] = perception_inferences.pop(0)

        return perception_trajectory

    def get_agent_state(self, target_agent, context):
        prompt = f"{context}\n\nQuestion: What are the facts regarding {target_agent} in the above context? Please only output the facts directly related to {target_agent} without any additional comments."
        state = self.tracer_model.interact(prompt, temperature=0)
        return state

    def preprocess_input(self, text: str, target_agent=None) -> dict:
        """
        Preprocess the input_text before passing it to the tracer model. Identify the target agent (i.e., character) and label the actions.
        """
        preprocessed_text = BaseTracer.preprocess_input(self, text, target_agent)
        if preprocessed_text is None:
            return None
        preprocessed_text['perceptions'] = self.track_perception(preprocessed_text['trajectory'], preprocessed_text['target_agent'])
        preprocessed_text['assumption'] = self.get_assumption(preprocessed_text['question'])
        self.assumption = f"\n{preprocessed_text['assumption']}"

        return preprocessed_text

    def initialize(self, state_action, perceptions):
        context_input = ""
        state, action = state_action['state'], state_action['action']
        if state:
            if self.args.input_is_chat:
                context_input += f"{state}\n"
            else:
                agent_state = self.get_agent_state(self.target_agent, state)
                context_input += f"<state>\n{agent_state.strip()}\n</state>\n"
            context_input += f"<note>{perceptions['state']}</note>\n\n"
        
        if action:
            if self.args.input_is_chat:
                context_input += f"<response>\n{action}\n</response>\n"
            else:
                context_input += f"<action>\n{action}\n</action>\n"

            if perceptions['action']:
                context_input += f"<note>{perceptions['action']}</note>\n"

        n_hypotheses_str = str(self.args.n_hypotheses)

        if self.args.n_hypotheses > 1:
            if action:
                belief_query = f"{context_input.strip()}{self.assumption}\n\nGenerate a numbered list of {n_hypotheses_str} hypotheses on what were {self.target_agent}'s thoughts (e.g., beliefs, intent) that led to the action above. Do not add any additional comments."
            else:
                belief_query = f"{context_input.strip()}{self.assumption}\n\nGenerate a numbered list of {n_hypotheses_str} hypotheses on what {self.target_agent} will be thinking (e.g., beliefs). Do not add any additional comments."
            _hypotheses_list = prompting_for_ordered_list(self.tracer_model, prompt=belief_query, n=self.args.n_hypotheses)
            hypotheses_list = [hypothesis.strip() for hypothesis in _hypotheses_list]
        else:
            belief_query = f"{context_input}\n\nQuestion: What will {self.target_agent} be thinking now?"
            hypothesis = self.tracer_model.interact(belief_query, temperature=0, max_tokens=1024)
            hypotheses_list = [hypothesis]

        weights = np.ones(len(hypotheses_list)) / len(hypotheses_list)
        initial_hypotheses = HypothesesSetV3(target_agent=self.target_agent, contexts=[state_action], perceptions=[perceptions], texts=hypotheses_list, weights=weights)

        return initial_hypotheses

    def get_assumption(self, question: str):
        if self.args.dataset == 'mmtom':
            if ", " in question:
                assumption_substring = question.split(", ")[0]
            else:
                return None
        else:
            return ""
        prompt = f"{assumption_substring}\n\nTask: Convert the above into a sentence in present tense. Do not add any additional comments."
        model = load_model("gpt-4o-2024-08-06", run_id=self.args.run_id) if self.args.use_helper_llm else self.tracer_model
        assumption = model.interact(prompt, temperature=0, max_tokens=128)
        if self.args.print:
            print(Panel(assumption, title="Assumption", style="green"))

        return assumption 

    def interleave_context_and_perception(self, context_history: List[dict], perception_history: List[dict], target_agent: str = None, chat: bool = False) -> str:
        if target_agent is None:
            target_agent = self.target_agent
        context_and_perception = ""
        for idx, (c, p) in enumerate(zip(context_history, perception_history)):
            if c['state'] or c['action']:
                context_and_perception += f"<context {idx + 1}>\n"
                if c['state']:
                    context_and_perception += f"<state>{p['state']}</state>\n\n" # only include the perception for inhibitory control
                if c['action']:
                    if self.args.input_is_chat:
                        context_and_perception += f"<response>\n{c['action']}\n</response>\n"
                    else:
                        context_and_perception += f"<action>{c['action']}</action>\n"

                    if p['action']:
                        context_and_perception += f"<note>{p['action']}</note>\n"
            context_and_perception = context_and_perception.strip() + f"\n</context {idx + 1}>\n\n"
        return context_and_perception.strip()

    def setup_propagation(self, existing_hypotheses: HypothesesSetV3, state_action: dict, perceptions:dict) -> HypothesesSetV3:
        target_agent = existing_hypotheses.target_agent
        context_history = deepcopy(existing_hypotheses.contexts)
        perception_history = deepcopy(existing_hypotheses.perceptions)
        context_and_perception_str = self.interleave_context_and_perception(context_history, perception_history, target_agent)
        context_history.append(state_action)
        perception_history.append(perceptions)

        new_context = ""
        if state_action['state']:
            if self.args.input_is_chat:
                new_context += f"<state>\n{state_action['state']}\n</state>\n"
            else:
                new_context += f"<state>{state_action['state']}</state>\n"
            if perceptions['state']:
                new_context += f"<note>{perceptions['state']}</note>\n\n"
        if state_action['action']:
            if self.args.input_is_chat:
                new_context += f"<response>\n{state_action['action']}\n</response>\n"
            else:
                new_context += f"<action>{state_action['action']}</action>\n"
            if perceptions['action']:
                new_context += f"<note>{perceptions['action']}</note>"

        return {"target_agent": target_agent, "context_and_perception_str": context_and_perception_str, "new_context": new_context.strip(), "context_history": context_history, "perception_history": perception_history}

    def weigh(self, hypotheses: HypothesesSetV3, action: str, mode: str) -> dict:
        # Weight the hypotheses based on the next action
        hypotheses_texts = hypotheses.texts
        target_agent = hypotheses.target_agent
        context_history = hypotheses.contexts
        perception_history = hypotheses.perceptions

        if mode == "prompting":
            results = self.prompt_likelihood(hypotheses_texts, context_history, perception_history, action)
        else:
            raise NotImplementedError

        return results

    def prompt_likelihood(self, existing_hypotheses: list, context_history: list, perception_history: list, action: str, target_agent: str = None):
        def map_response_to_score(response: str, mapping: dict) -> float:
            # check whether the response is among the keys of the mapping
            for option in mapping.keys():
                if response.startswith(f"({option})") or response.startswith(f"{option})") or response.startswith(f"{option}.") or response.startswith(f"{option} ") or f"({option})" in response or f" {option})" in response or f" {option}." in response or option == response:
                    return mapping[option]
            return 0.001

        target_agent = self.target_agent if target_agent is None else target_agent
        # TODO: maybe trim the last action and its perception because the thought actually comes before the action.
        pruned_context_history = deepcopy(context_history)
        pruned_perception_history = deepcopy(perception_history)
        # to remove the last action and its perception, because we will be evaluating the thought before the action
        pruned_context_history[-1]['action'] = None
        pruned_perception_history[-1]['action'] = None 
        if len(pruned_context_history) > 1:
            context_and_perception_str = self.interleave_context_and_perception(pruned_context_history, pruned_perception_history)
        else:
            if pruned_context_history[-1]['state']:
                context_and_perception_str = f"{pruned_context_history[-1]['state']}"
            else:
                context_and_perception_str = ""

        system_prompt = f"Your job is to evaluate the probability of actions/utterance under a given fact. Use common sense: for instance, if someone is searching for an item, they are likely to take it once they find it rather than merely observing it. If they don't take it and just sees it, it indicates a lack of interest and that was not what they were looking for. Briefly explain the probability of the action/utterance under the given fact first and then give the answer option with prefix 'Answer:'"
        question = f"Question: Based on the context and {target_agent}'s thoughts provided, would {target_agent} do the next actions or say the next utterances described above? Let's think step by step and give the final answer."
        word_mapping = {'a': "Very Likely (Around 90%)", 'b': "Likely (Around 70%)", 'c': "Somewhat Likely (Around 60%)", 'd': "Somewhat Unlikely (Around 25%)", 'e': "Unlikely (Around 20%)", 'f': "Very Unlikely (Below 10%)"}
        score_mapping = {'a': 3, 'b': 2.5, 'c': 2, 'd': 1, 'e': 0.5, 'f': 0.001}
        multiple_choice_options = ""
        for k, v in word_mapping.items():
            multiple_choice_options += f"({k}) {v}\n"
        if self.args.input_is_chat:
            likelihood_prompts = [f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n<{target_agent}'s thoughts>\n{hypothesis}\n</{target_agent}'s thoughts>\n\n<response>\n{action}\n</response>\n\n{question}\n{multiple_choice_options}" for hypothesis in existing_hypotheses]
        else:
            likelihood_prompts = [f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n<{target_agent}'s thoughts>\n{hypothesis}\n</{target_agent}'s thoughts>\n\n<next action>{action}</next action>\n<note>{perception_history[-1]['action']}</note>\n\n{question}\n{multiple_choice_options}" for hypothesis in existing_hypotheses]
        raw_predictions = self.tracer_model.batch_interact(likelihood_prompts, temperature=0, system_prompts=system_prompt, max_tokens=512)
        reasonings = []
        answers = []
        for response in raw_predictions:
            reasoning, a = response.split("Answer:")[0].strip(), response.split("Answer:")[-1].strip()
            reasonings.append(reasoning)
            answers.append(a)
        raw_scores = np.array([map_response_to_score(j, score_mapping) for j in answers])
        weights = softmax(raw_scores)

        results = {
            'prompts': likelihood_prompts,
            'raw_predictions': raw_predictions,
            'reasonings': reasonings,
            'raw_scores': raw_scores,
            'weights': weights
        }

        return results

    def propagate(self, existing_hypotheses: HypothesesSetV3, state_action: dict, perceptions: dict) -> HypothesesSetV3:
        """
        Propagate the hypotheses on the target agent using the context, which is the text that does not contain the target agent's actions.
        """
        prop_info = self.setup_propagation(existing_hypotheses, state_action, perceptions)
        target_agent = prop_info["target_agent"]
        context_history = prop_info["context_history"]
        perception_history = prop_info["perception_history"]
        context_and_perception_str = prop_info["context_and_perception_str"]
        new_context = prop_info["new_context"]

        system_prompt = f"You are an expert assistant trying to predict {target_agent}'s thoughts."
        propagation_prompts = [f"{self.trace_header}\n\n<previous context>\n{context_and_perception_str}\n</previous context>\n<previous prediction regarding {target_agent}'s thoughts>\n{hypothesis}\n</previous prediction regarding {target_agent}'s thoughts>\n\n<current context>{self.assumption}\n{new_context}\n</current context>\n\nQuestion: What did {target_agent} believe?" for hypothesis in existing_hypotheses.texts]
        propagated_texts = self.tracer_model.batch_interact(propagation_prompts, system_prompts=system_prompt, temperature=0, max_tokens=1024)
        propagated_hypotheses = HypothesesSetV3(target_agent, context_history, perception_history, propagated_texts, existing_hypotheses.weights, parent_hypotheses=existing_hypotheses.hypotheses)

        return propagated_hypotheses

    def rejuvenate_hypotheses(self, existing_hypotheses: HypothesesSetV3) -> HypothesesSetV3:
        """
        Rejuvenate hypotheses by paraphrasing the hypotheses
        """
        for h in existing_hypotheses.texts:
            print(Panel(h, title="Low Variance Hypotheses", style="red", box=box.SIMPLE_HEAD))

        if not self.args.use_perception_only:
            system_prompt = f"Your task is to paraphrase the following text. Make sure to keep the meaning of the text intact while rephrasing them. Do not add any additional comments."
            revision_prompts = [f"{hypothesis}" for hypothesis in existing_hypotheses.texts]
            revised_texts = self.tracer_model.batch_interact(revision_prompts, system_prompts=system_prompt, temperature=1, max_tokens=1024)
            existing_hypotheses.texts = revised_texts
        overall_text_diversity = 1 - overall_jaccard_similarity(existing_hypotheses.texts)
        print(Panel(f"Text diversity: {overall_text_diversity}", title="Diversity of the Jittered Hypotheses", style="blue", box=box.SIMPLE_HEAD))
        print(Panel("\n".join(existing_hypotheses.texts), title="Jittered hypotheses", style="blue", box=box.SIMPLE_HEAD))

        return existing_hypotheses

    def weighted_average_hypotheses(self, hypotheses: HypothesesSetV3, top_p: float = 0.9) -> dict:
        """
        Weighted mean estimate of the hypotheses.
        """
        target_agent = hypotheses.target_agent
        sorted_hypotheses = sorted(zip(hypotheses.texts, hypotheses.weights), key=lambda x: x[1], reverse=True)

        # only select up to cumulative weights of top_p
        top_hypotheses = []
        cumulative_weight = 0
        for hypothesis, weight in sorted_hypotheses:
            top_hypotheses.append((hypothesis, weight))
            cumulative_weight += weight
            if cumulative_weight >= top_p:
                break

        context_history = deepcopy(hypotheses.contexts)
        perception_history = deepcopy(hypotheses.perceptions)
        context_and_perception_str = self.interleave_context_and_perception(context_history, perception_history, target_agent)  
        hypotheses_str = f"{context_and_perception_str}\n\n<{target_agent}'s thoughts>\n"
        for idx, (hypothesis, weight) in enumerate(top_hypotheses):
            hypotheses_str += f"**Prediction {str(idx + 1)} (Weight: {weight:.2f}):**\n{hypothesis}\n\n\n"
        hypotheses_str = hypotheses_str.strip()
        hypotheses_str += f"\n</{target_agent}'s thoughts>\n\nQuestion: What did {target_agent} believe?"

        aggregated_hypothesis = self.tracer_model.interact(hypotheses_str, max_tokens=1234)
        final_hypothesis = f"<{target_agent}'s updated thoughts>\n{aggregated_hypothesis}\n</{target_agent}'s updated thoughts>"

        return {'text': final_hypothesis, 'likelihood': list(hypotheses.weights), 'aggregated': True, 'context': hypotheses.contexts[-1], 'perception': hypotheses.perceptions[-1], 'hypothesis': aggregated_hypothesis}

    def chain_weighted_average_trace(self, hypotheses_list: List[HypothesesSetV3]) -> dict:
        """
        Chain the trace of hypotheses using weighted average

        Args:
            hypotheses (HypothesesSet): 

        Returns:
            str: a summary of the hypothesis trace
        """
        target_agent = self.target_agent
        averaged_hypotheses_list = [self.weighted_average_hypotheses(hypotheses) for hypotheses in hypotheses_list]

        trace_str = ""
        for idx, h in enumerate(averaged_hypotheses_list):
            context_str = ""
            if h['context']['state']:
                context_str += f"{h['context']['state']}\n" # newly added
                context_str += f"<note>{h['perception']['state']}</note>\n"
            if h['context']['action']:
                context_str += f"{h['context']['action']}\n"
                if h['perception']['action']:
                    context_str += f"<note>{h['perception']['action']}</note>"
            trace_str += f"<context {str(idx + 1)}>\n{context_str.strip()}\n\n<{target_agent}'s updated thoughts>{h['hypothesis']}</{target_agent}'s updated thoughts>\n</context {str(idx + 1)}>\n\n"
            
        return {'text': trace_str.strip(), 'aggregated': True}

    def _trace(self, text: str, target_agent=None):
        preprocessed_text = self.preprocess_input(text, target_agent)
        if preprocessed_text is None:
            print(cf.bold | cf.magenta("Failed to identify the target agent."))
            self.dump({'summary': ""}, [])
            return ""
        self.set_tracer_variables(preprocessed_text)

        trajectory = preprocessed_text['trajectory']
        perceptions_trajectory = preprocessed_text['perceptions']

        hypotheses_list = []
        context_history = []
        for idx, (state_action, perceptions) in enumerate(zip(trajectory, perceptions_trajectory)):
            if idx == 0:
                new_hypotheses = self.initialize(state_action=state_action, perceptions=perceptions)
            else:
                existing_hypotheses = hypotheses_list[-1]
                new_hypotheses = self.propagate(existing_hypotheses, state_action=state_action, perceptions=perceptions)

            if state_action['action']:
                weight_results = self.weigh(new_hypotheses, state_action['action'], mode="prompting")
                new_hypotheses.update_weights(weight_results['weights'])
                new_hypotheses.weight_details = weight_results

                # resample hypotheses or jitter
                if self.args.n_hypotheses > 1:
                    ess = compute_ess(new_hypotheses)
                    overall_text_diversity = 1 - overall_jaccard_similarity(new_hypotheses.texts)
                    if ess < (self.args.n_hypotheses) / 2:
                        new_hypotheses = resample_hypotheses_with_other_info(new_hypotheses, ess)
                    elif overall_text_diversity < 0.25:
                        print(Panel(f"Text diversity: {overall_text_diversity}", title="Low Variance Hypotheses", style="red"))
                        new_hypotheses = self.rejuvenate_hypotheses(new_hypotheses)
            else:
                pass

            hypotheses_list.append(new_hypotheses)

            # update history
            if state_action['state']:
                context_history.append({'text': state_action['state'], 'action': False})
            if state_action['action']:
                context_history.append({'text': state_action['action'], 'action': True})

        traced_thoughts = self.chain_weighted_average_trace(hypotheses_list)
        trace_text = f"{self.trace_header}\n\n{traced_thoughts['text']}"

        self.dump(traced_thoughts, hypotheses_list)
        return trace_text

    def trace(self, input_text, target_agent=None):
        # check if the input text, target_character is cached 
        if target_agent is None:
            target_agent = self.identify_target(input_text)
        if self.args.dataset != "mmtom":
            context = input_text.split("\nQuestion:")[0].strip()
        else:
            context = input_text
        if self.args.dataset == "fantom":
            context = context.split("\n\nTarget:")[0].strip()
            context = context.split("\n\nInformation:")[0].strip()
        elif self.args.dataset == "confaide":
            context = context.split("Meeting:")[-1].strip()

        # check if the input text is already traced in self.cache_db
        if context in self.cache_db and target_agent in self.cache_db[context]:
            print(Panel(f">>> Using cached trace for {target_agent}!", style="yellow"))
            return self.cache_db[context][target_agent]
        
        print(Panel(f">>> Tracing {target_agent}'s thoughts!", style="blue"))
        trace_result = self._trace(context, target_agent)

        # cache the trace result for the same input text and character
        if context not in self.cache_db:
            self.cache_db[context] = {}
            self.cache_db[context][target_agent] = trace_result
        else:
            self.cache_db[context][target_agent] = trace_result

        return trace_result

class MultiTracer(Tracer):
    def identify_target(self, input_text: str) -> str:
        """
        Identify the target agent that we have to trace by looking at the question.

        Args:
            input_text (str): The context text.

        Returns:
            str: The target agent.
        """
        question = extract_question(input_text)
        target_identification_prompt = f"'{question}'\n\nMain question: Who is the subject of the above question? Whose perspective is this question primarily about? Provide the names of the individual, their title, or the group. If the subject of the question is not related to a person or a group, state 'none'. If the question is asking to list the names, state 'all'.\nThe concise answer to the main question is (e.g, name):"

        if self.args.use_helper_llm:
            llm = load_model('gpt-4o-2024-11-20', run_id=self.args.run_id)
            output = llm.interact(target_identification_prompt, temperature=0, max_tokens=16)
        else:
            output = self.tracer_model.interact(target_identification_prompt, temperature=0, max_tokens=16)
        target_agent = output.split("\n")[0].split(":")[-1].strip().strip(".").replace("*", "")

        if target_agent == 'all':
            print(Panel(f"Identified target agent as 'all'!", style="yellow"))
            context = input_text.split(question)[0].split("\n\n")[0]
            target_identification_prompt = f"{context}\n\nName all the people or groups who are in the context. Use commas. Do not include any additional comments."
            if self.args.use_helper_llm:
                llm = load_model('gpt-4o-2024-11-20', run_id=self.args.run_id)
                output = llm.interact(target_identification_prompt, temperature=0, max_tokens=16)
            else:
                output = self.tracer_model.interact(target_identification_prompt, temperature=0, max_tokens=16)
            target_agent = output

        return target_agent

    def trace(self, input_text, target_agent=None):
        # check if the input text, target_character is cached 
        if target_agent is None:
            target_agent = self.identify_target(input_text)
        context = input_text.split("\nQuestion:")[0].strip()
        if self.args.dataset == 'fantom':
            context = context.split("\n\nTarget:")[0].strip()
            context = context.split("\n\nInformation:")[0].strip()

        if "," in target_agent:
            _target_agents = target_agent.split(",")
            target_agents = [a.strip() for a in _target_agents]
            trace_result = ""
            for idx, ta in enumerate(target_agents):
                if context in self.cache_db and ta in self.cache_db[context]:
                    print(Panel(f">>> Using cached trace for {ta}!", style="yellow"))
                    individual_trace = self.cache_db[context][ta]
                else:
                    print(Panel(f">>> Tracing {ta}'s thoughts!", style="blue"))
                    individual_trace = self._trace(context, ta)
                if idx == 0:
                    _individual_trace = individual_trace.replace("To answer this question,", "To answer this question, first,").strip()
                elif idx > 0 and idx < len(target_agents) - 1:
                    _individual_trace = individual_trace.replace("To answer this question", "Next").strip()
                else:
                    _individual_trace = individual_trace.replace("To answer this question", "Finally").strip()
                trace_result += _individual_trace + "\n\n"

                # cache the trace result for the same input text and character
                if context not in self.cache_db:
                    self.cache_db[context] = {}
                    self.cache_db[context][ta] = individual_trace
                else:
                    self.cache_db[context][ta] = individual_trace
        else:
            # check if the input text is already traced in self.cache_db
            if context in self.cache_db and target_agent in self.cache_db[context]:
                print(Panel(f">>> Using cached trace for {target_agent}!", style="yellow"))
                return self.cache_db[context][target_agent]
            
            print(Panel(f">>> Tracing {target_agent}'s thoughts!", style="blue"))
            trace_result = self._trace(context, target_agent)

            # cache the trace result for the same input text and character
            if context not in self.cache_db:
                self.cache_db[context] = {}
                self.cache_db[context][target_agent] = trace_result
            else:
                self.cache_db[context][target_agent] = trace_result

        return trace_result.strip()

class TracerLight(Tracer):
    """
    TracerLight is a simplified version of Tracer that does propagation and likelihood calculation of all hypotheses at once in a single prompt.
    This leads to much worse performance.
    """
    def initialize(self, state_action, perceptions):
        context_input = ""
        state, action = state_action['state'], state_action['action']
        if state:
            if self.args.input_is_chat:
                context_input += f"{state}\n"
            else:
                agent_state = self.get_agent_state(self.target_agent, state)
                context_input += f"<state>\n{agent_state.strip()}\n</state>\n"
            context_input += f"<note>{perceptions['state']}</note>\n\n"

        n_hypotheses_str = str(self.args.n_hypotheses)

        if self.args.n_hypotheses > 1:
            if action:
                belief_query = f"{context_input.strip()}{self.assumption}\n\nGenerate a numbered list of {n_hypotheses_str} hypotheses on what were {self.target_agent}'s thoughts (e.g., beliefs, intent) that led to the action above. Do not add any additional comments."
            else:
                belief_query = f"{context_input.strip()}{self.assumption}\n\nGenerate a numbered list of {n_hypotheses_str} hypotheses on what {self.target_agent} will be thinking (e.g., beliefs). Do not add any additional comments."
            _hypotheses_list = prompting_for_ordered_list(self.tracer_model, prompt=belief_query, n=self.args.n_hypotheses)
            hypotheses_list = [hypothesis.strip() for hypothesis in _hypotheses_list]
        else:
            belief_query = f"{context_input}\n\nQuestion: What will {self.target_agent} be thinking now?"
            hypothesis = self.tracer_model.interact(belief_query, temperature=0, max_tokens=1024)
            hypotheses_list = [hypothesis]

        weights = np.ones(len(hypotheses_list)) / len(hypotheses_list)
        initial_hypotheses = HypothesesSetV3(target_agent=self.target_agent, contexts=[state_action], perceptions=[perceptions], texts=hypotheses_list, weights=weights)

        return initial_hypotheses

    def propagate(self, existing_hypotheses: HypothesesSetV3, state_action: dict, perceptions: dict) -> HypothesesSetV3:
        """
        Propagate the hypotheses on the target agent using the context, which is the text that does not contain the target agent's actions.
        """

        prop_info = self.setup_propagation(existing_hypotheses, state_action, perceptions) # get interleaved context and perception
        target_agent = prop_info["target_agent"]
        context_history = prop_info["context_history"]
        perception_history = prop_info["perception_history"]
        context_and_perception_str = prop_info["context_and_perception_str"]
        new_context = prop_info["new_context"]

        system_prompt = f"You are an expert assistant trying to predict {target_agent}'s thoughts. Update the previous {self.args.n_hypotheses} predictions based on the new context and the new action. Try to make the predictions diverse to cover a wide range of possibilities even low probability ones. Output {self.args.n_hypotheses} updated predictions in an ordered list. Do not add any additional comments."
        ordered_hypotheses_string = '\n'.join([f"{index + 1}. {item}" for index, item in enumerate(existing_hypotheses.texts)])

        propagation_prompt = f"{self.trace_header}\n\n<previous context>\n{context_and_perception_str}\n</previous context>\n<previous predictions regarding {target_agent}'s thoughts>\n{ordered_hypotheses_string}\n</previous predictions regarding {target_agent}'s thoughts>\n\n<current context>{self.assumption}\n{new_context}\n</current context>\n\nQuestion: What did {target_agent} believe?"
        temperature = 0
        while True:
            propagated_text = self.tracer_model.interact(propagation_prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=2024)
            propagated_hypotheses_str_list = capture_and_parse_ordered_list(propagated_text)
            if len(propagated_hypotheses_str_list) == self.args.n_hypotheses:
                break
            else:
                print(Panel(f"Number of propagated hypotheses is not equal to {self.args.n_hypotheses}! Retrying...", box=box.SIMPLE_HEAD, style="red"))
                temperature += 0.3
                if temperature > 1.2:
                    break

        propagated_hypotheses = HypothesesSetV3(target_agent, context_history, perception_history, propagated_hypotheses_str_list, existing_hypotheses.weights, parent_hypotheses=existing_hypotheses.hypotheses)

        return propagated_hypotheses

    def prompt_likelihood(self, existing_hypotheses: list, context_history: list, perception_history: list, action: str, target_agent: str = None):

        target_agent = self.target_agent if target_agent is None else target_agent
        # TODO: maybe trim the last action and its perception because the thought actually comes before the action.
        pruned_context_history = deepcopy(context_history)
        pruned_perception_history = deepcopy(perception_history)
        if len(pruned_context_history) > 1:
            context_and_perception_str = self.interleave_context_and_perception(pruned_context_history[:-1], pruned_perception_history[:-1]) # to remove the last action and its perception, because we will be evaluating the thought before the action
        else:
            if pruned_context_history[-1]['state']:
                context_and_perception_str = f"{pruned_context_history[-1]['state']}"
            else:
                context_and_perception_str = ""

        system_prompt = f"Your job is to rate the probability (0-1) of actions/utterance under a list of given different hypotheses. Use common sense: for instance, if someone is searching for an item, they are likely to take it once they find it rather than merely observing it. If they don't take it and just sees it, it indicates a lack of interest and that was not what they were looking for. For each hypothesis, briefly explain the probability of the action/utterance under each hypothesis first. Then, at the end of your response, aggregate the answer for each hypothesis in a simple ordered list with prefix 'Final Answer:'"
        question = f"Question: Rate the probability (0-1) of the <next action> or <next response> described above under each given hypothesis. Let's think step by step and give the final answer."
        hypothesis_str = '\n'.join([f"Hypothesis {index + 1}. {item}" for index, item in enumerate(existing_hypotheses)])
        if self.args.input_is_chat:
            likelihood_prompt = f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n<{target_agent}'s thoughts>\n{hypothesis_str}\n</{target_agent}'s thoughts>\n\n<next response>\n{action}\n</next response>\n\n{question}"
        else:
            likelihood_prompt = f"<previous context>\n{context_and_perception_str}\n</previous context>\n\n<{target_agent}'s thoughts>\n{hypothesis_str}\n</{target_agent}'s thoughts>\n\n<next action>{action}</next action>\n<note>{perception_history[-1]['action']}</note>\n\n{question}"
        raw_predictions = self.tracer_model.interact(likelihood_prompt, temperature=0, system_prompt=system_prompt, max_tokens=2048)
        if "Final Answer:" in raw_predictions:
            reasoning, answer = raw_predictions.split("Final Answer:")[0], raw_predictions.split("Final Answer:")[-1]
        elif "Final Answer**" in raw_predictions:
            reasoning, answer = raw_predictions.split("Final Answer**")[0], raw_predictions.split("Final Answer**")[-1]
        else:
            reasoning = answer = raw_predictions
        probs = capture_and_parse_ordered_list(answer)
        converted_probs = [float(prob.split(":")[-1].strip().strip("*").strip()) for prob in probs]

        # normalize the probabilities to sum to 1
        converted_probs = converted_probs / np.sum(converted_probs)
        weights = converted_probs

        results = {
            'prompts': [likelihood_prompt] * len(existing_hypotheses),
            'raw_predictions': raw_predictions,
            'reasonings': reasoning,
            'raw_scores': probs,
            'weights': weights
        }

        return results

    def rejuvenate_hypotheses(self, existing_hypotheses: HypothesesSetV3) -> HypothesesSetV3:
        """
        Rejuvenate hypotheses by paraphrasing the hypotheses
        """
        for h in existing_hypotheses.texts:
            print(Panel(h, title="Low Variance Hypotheses", style="red", box=box.SIMPLE_HEAD))

        system_prompt = f"Your task is to paraphrase the following list of texts. Make sure to keep the meaning of the texts intact while rephrasing them. If there are identical texts, try to make them slightly different. Do not add any additional comments and output the revised texts in an ordered list."
        hypotheses_list = [f"{index + 1}. {item}" for index, item in enumerate(existing_hypotheses.texts)]
        hypotheses_list_str = "\n".join(hypotheses_list)
        revised_text = self.tracer_model.interact(hypotheses_list_str, system_prompt=system_prompt, temperature=1, max_tokens=1024)
        revised_hypotheses_list = capture_and_parse_ordered_list(revised_text)
        existing_hypotheses.texts = revised_hypotheses_list
        overall_text_diversity = 1 - overall_jaccard_similarity(existing_hypotheses.texts)
        print(Panel(f"Text diversity: {overall_text_diversity}", title="Diversity of the Jittered Hypotheses", style="blue", box=box.SIMPLE_HEAD))
        print(Panel("\n".join(existing_hypotheses.texts), title="Jittered hypotheses", style="blue", box=box.SIMPLE_HEAD))

        return existing_hypotheses

    def weighted_average_hypotheses(self, hypotheses: HypothesesSetV3, top_p: float = 0.9) -> dict:
        """
        Weighted mean estimate of the hypotheses.
        """
        
        target_agent = hypotheses.target_agent
        sorted_hypotheses = sorted(zip(hypotheses.texts, hypotheses.weights), key=lambda x: x[1], reverse=True)

        # only select up to cumulative weights of top_p
        top_hypotheses = []
        cumulative_weight = 0
        for hypothesis, weight in sorted_hypotheses:
            top_hypotheses.append((hypothesis, weight))
            cumulative_weight += weight
            if cumulative_weight >= top_p:
                break

        hypotheses_str = ""
        for idx, (hypothesis, weight) in enumerate(top_hypotheses):
            hypotheses_str += f"**Prediction {str(idx + 1)} (Weight: {weight:.2f}):**\n{hypothesis}\n\n"
        hypotheses_str = hypotheses_str.strip()
        final_hypothesis = f"<{target_agent}'s updated thoughts>\n{hypotheses_str}\n</{target_agent}'s updated thoughts>"

        return {'text': final_hypothesis, 'likelihood': list(hypotheses.weights), 'aggregated': True, 'context': hypotheses.contexts[-1], 'perception': hypotheses.perceptions[-1], 'hypothesis': hypotheses_str}

    def chain_weighted_average_trace(self, hypotheses_set_list: List[HypothesesSetV3]) -> dict:
        """
        Chain the trace of hypotheses using weighted average

        Args:
            hypotheses (HypothesesSet): 

        Returns:
            str: a summary of the hypothesis trace
        """
        target_agent = self.target_agent
        averaged_hypotheses_list = [self.weighted_average_hypotheses(hypotheses) for hypotheses in hypotheses_set_list]

        trace_str = ""
        for idx, h in enumerate(averaged_hypotheses_list):
            context_str = ""
            if h['context']['state']:
                context_str += f"{h['context']['state']}\n" # newly added
                context_str += f"<note>{h['perception']['state']}</note>\n"
            if h['context']['action']:
                context_str += f"{h['context']['action']}\n"
                if h['perception']['action']:
                    context_str += f"<note>{h['perception']['action']}</note>"
            trace_str += f"<context {str(idx + 1)}>\n{context_str.strip()}\n\n<{target_agent}'s updated thoughts>\n{h['hypothesis']}</{target_agent}'s updated thoughts>\n</context {str(idx + 1)}>\n\n"
            
        return {'text': trace_str.strip(), 'aggregated': True}

    def _trace(self, text: str, target_agent=None):
        preprocessed_text = self.preprocess_input(text, target_agent)
        if preprocessed_text is None:
            print(cf.bold | cf.magenta("Failed to identify the target agent."))
            self.dump({'summary': ""}, [])
            return ""
        self.set_tracer_variables(preprocessed_text)

        trajectory = preprocessed_text['trajectory']
        perceptions_trajectory = preprocessed_text['perceptions']

        hypotheses_list = []
        context_history = []
        for idx, (state_action, perceptions) in enumerate(zip(trajectory, perceptions_trajectory)):
            if idx == 0:
                new_hypotheses = self.initialize(state_action=state_action, perceptions=perceptions)
            else:
                existing_hypotheses = hypotheses_list[-1]
                new_hypotheses = self.propagate(existing_hypotheses, state_action=state_action, perceptions=perceptions)

            if state_action['action']:
                weight_results = self.weigh(new_hypotheses, state_action['action'], mode="prompting")
                new_hypotheses.update_weights(weight_results['weights'])
                new_hypotheses.weight_details = weight_results

                # resample hypotheses or jitter
                if self.args.n_hypotheses > 1:
                    ess = compute_ess(new_hypotheses)
                    overall_text_diversity = 1 - overall_jaccard_similarity(new_hypotheses.texts)
                    if ess < (self.args.n_hypotheses) / 2:
                        new_hypotheses = resample_hypotheses_with_other_info(new_hypotheses, ess)
                    elif overall_text_diversity < 0.25:
                        print(Panel(f"Text diversity: {overall_text_diversity}", title="Low Variance Hypotheses", style="red"))
                        new_hypotheses = self.rejuvenate_hypotheses(new_hypotheses)
            else:
                pass

            hypotheses_list.append(new_hypotheses)

            # update history
            if state_action['state']:
                context_history.append({'text': state_action['state'], 'action': False})
            if state_action['action']:
                context_history.append({'text': state_action['action'], 'action': True})

        traced_thoughts = self.chain_weighted_average_trace(hypotheses_list)
        trace_text = f"{self.trace_header}\n\n{traced_thoughts['text']}"

        self.dump(traced_thoughts, hypotheses_list)
        return trace_text

class MultiTracerLight(TracerLight, MultiTracer):
    pass

def load_tracer_model(args):
    if args.tracer_type == 'tracer':
        tracing_model = Tracer(args)
    elif args.tracer_type == 'multi-tracer':
        tracing_model = MultiTracer(args)
    elif args.tracer_type == 'tracer-light':
        tracing_model = TracerLight(args)
    elif args.tracer_type == 'multi-tracer-light':
        tracing_model = MultiTracerLight(args)
    else:
        raise NotImplementedError

    return tracing_model