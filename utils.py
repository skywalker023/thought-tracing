import os
import re
import json
import pandas as pd
import numpy as np
from typing import List

from nltk.stem import LancasterStemmer

from agents.base import BaseAgent

PROJECT_BASE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(PROJECT_BASE, 'prompt_templates')

def log_outputs(outputs, dataset, model, tracing_model="none", particle="0", **kwargs):
    outputs_dir = os.path.join(PROJECT_BASE, "outputs", dataset, kwargs['condition'])
    os.makedirs(outputs_dir, exist_ok=True)

    # log outputs in json lines format
    model_name = model.split("/")[-1]
    tracing_model_name = tracing_model.split("/")[-1]
    output_file = os.path.join(outputs_dir, f"model-{model_name}_tracer-{tracing_model_name}_particle-n-{particle}_{dataset}.jsonl")

    try:
        with open(output_file, 'a') as f:
            for output in outputs:
                f.write(json.dumps(output) + '\n')
    except:
        print("Error writing to file")

def read_logs(dataset, model, tracing_model="none", particle="0", **kwargs):
    outputs_dir = os.path.join(PROJECT_BASE, "outputs", dataset, kwargs['condition'])
    model_name = model.split("/")[-1]
    tracing_model_name = tracing_model.split("/")[-1]
    output_file = os.path.join(outputs_dir, f"model-{model_name}_tracer-{tracing_model_name}_particle-n-{particle}_{dataset}.jsonl")

    if os.path.exists(output_file):
        logs = pd.read_json(output_file, lines=True)
        return logs
    else:
        return None

def load_prompt(path: str):
    if path is None:
        return None
    
    template_path = os.path.join(TEMPLATE_DIR, path)
    with open(template_path, 'r') as file:
        prompt_template = file.read()
    return prompt_template

def check_list_formatting(text: str) -> str:
    """
    Check whether a given text is in an ordered list or an unordered list format.

    Args:
        text (_type_): _description_

    Returns:
        str: _description_
    """
    lines = text.strip().split('\n')
    # TODO: Update this to catch if there's any numbered list in the text, or if there's any unordered list in the text.
    for line in lines:
        if re.match(r'^\d+\.\s', line):
            return "ordered"
        elif re.match(r'^[\*\-]\s', line):
            return "unordered"
        else:
            return "not_list"

def parse_kv_ordered_list(text: str) -> dict:
    """
    Args:
        text (_type_): 1. key: value\n2. key: value\n3. key: value\n

    Returns:
        dict: {1: {key: value}, 2: {key: value}, 3: {key: value}}
    """
    result_dict = {}
    lines = text.strip().split('\n')
    for line in lines:
        line = line.replace("*", "") # Remove all asterisks, if any. Models tend to output markdown bold text with asterisks.
        match = re.match(r'^(\d+)\.\s(.+)$', line.strip())
        if match:
            level = int(match.group(1))
            key_and_value = match.group(2)
            key, value = key_and_value.split(":")
            result_dict[level] = {key.strip(): value.strip()}
    return result_dict

def parse_unordered_list(text: str) -> list:
    """
    Args:
        text (_type_): * value\n* value\n* value\n or - value\n- value\n- value\n

    Returns:
        list: [value1, value2, value3]
    """
    result_list = []
    lines = text.strip().split('\n')
    for line in lines:
        match = re.match(r'^[\*\-]\s(.+)$', line.strip())
        if match:
            result_list.append(match.group(1))
    return result_list

def parse_ordered_list(text: str) -> list:
    """
    Args:
        text (_type_): 1. value\n2. value\n3. value\n

    Returns:
        list: [value1, value2, value3]
    """
    result_list = []
    lines = text.strip().split('\n')
    for line in lines:
        line = line.replace("*", "") # Remove all asterisks, if any. Models tend to output markdown bold text with asterisks.
        match = re.match(r'^\d+\.\s(.+)$', line.strip())
        if match:
            result_list.append(match.group(1))
    return result_list

def capture_and_parse_ordered_list(text: str):
    # Regular expression to match ordered list items (numbers followed by a period and a space)
    ordered_list_pattern = re.compile(r'(\d+\.\s+.+?)(?=\d+\.\s|$)', re.DOTALL)
    
    # Find all matches
    matches = ordered_list_pattern.findall(text)
    
    # Remove numbers from the beginning of each match
    ordered_list = [re.sub(r'^\d+\.\s+', '', match).strip() for match in matches]
    
    return ordered_list

def capture_and_parse_unordered_list(text: str):
    # Regular expression to match unordered list items (bullet point followed by a space)
    unordered_list_pattern = re.compile(r'([\*\-]\s+.+?)(?=[\*\-]\s|$)', re.DOTALL)
    
    # Find all matches
    matches = unordered_list_pattern.findall(text)
    
    # Remove bullet points from the beginning of each match
    unordered_list = [re.sub(r'^[\*\-]\s+', '', match).strip() for match in matches]
    
    return unordered_list

def parse_markdown_list(text: str) -> list:
    list_format = check_list_formatting(text)
    if list_format == "ordered":
        return parse_ordered_list(text)
    elif list_format == "unordered":
        return parse_unordered_list(text)
    else:
        return None

def prompting_for_ordered_list(model: BaseAgent, prompt: str, n: int, history: List = None, system_prompt: str = None) -> List[str]:
    tolerance = 3
    temperature = 0
    while tolerance > 0:
        output = model.interact(prompt, max_tokens=1024, temperature=temperature, history=history, system_prompt=system_prompt)
        parsed_list = capture_and_parse_ordered_list(output)
        if len(parsed_list) >= n:
            parsed_list = parsed_list[:n]
            break
        else:
            temperature += 0.33
            tolerance -= 1
    return parsed_list

def parse_yes_partially_no(text: str) -> str:
    if text.lower().startswith("yes"):
        return "yes"
    elif text.lower().startswith("partially"):
        return "partially"
    else:
        return "no"

def extract_json_from_string(input_string):
    """
    Extracts and parses JSON data from a string that contains plain text as well as JSON.

    Args:
        input_string (str): The string containing both plain text and JSON.

    Returns:
        dict: Parsed JSON data as a Python dictionary, or None if no valid JSON is found.
    """
    # Use regular expression to find JSON object in the string
    json_pattern = r'\{.*\}'  # Pattern for JSON-like structure (basic version)
    
    # Search for JSON pattern
    match = re.search(json_pattern, input_string, re.DOTALL)  # re.DOTALL allows '.' to match newlines
    
    if match:
        json_str = match.group(0)
        try:
            # Parse JSON string into a Python dictionary
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format.")
            return None
    else:
        print("No JSON object found in the string.")
        return None

def list_to_unordered_list_string(items, list_bullet="-"):
    """
    Converts a list of items into a string formatted as an unordered list.

    Args:
        items (list): A list of items to be converted into an unordered list.

    Returns:
        str: A string formatted as an unordered list.
    """
    # Convert list items to strings and format as an unordered list
    unordered_list = "\n".join([f"{list_bullet} {item}".strip() for item in items])
    
    return unordered_list

def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

def map_int_to_string(self, int_number):
    mapping = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty"}
    return mapping[int_number]

def jaccard_similarity(sent1, sent2):
    lancaster = LancasterStemmer()
    _tokens1 = sent1.lower().split()
    _tokens2 = sent2.lower().split()
    tokens1 = set([lancaster.stem(token) for token in _tokens1])
    tokens2 = set([lancaster.stem(token) for token in _tokens2])
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)

def overall_jaccard_similarity(sentences):
    n = len(sentences)
    total_similarity = 0.0
    pair_count = 0
    
    for i in range(n):
        for j in range(i + 1, n):  # j starts at i + 1 to skip self-similarity and redundant pairs
            total_similarity += jaccard_similarity(sentences[i], sentences[j])
            pair_count += 1
    
    # If pair_count is 0 (e.g., when n=1), avoid division by zero by returning 1 (self-similarity)
    return total_similarity / pair_count if pair_count > 0 else 1.0

def uncapitalize(s):
    return s[:1].lower() + s[1:]

def find_first_integer(s):
    # Use a regular expression to search for integers in the string
    match = re.search(r'\d+', s)
    
    # If a match is found, return it as an integer
    if match:
        return int(match.group())
    
    # If no match is found, return None or an appropriate message
    return 50

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)