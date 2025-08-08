import io
import os
import json
import base64
import pickle
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
from openai import OpenAI
random.seed(42)

import sys
sys.path.append("..")
from agents.load_model import load_model
from tracer import load_tracer_model, get_tracer_parser

from rich import print
from rich.panel import Panel
from rich import box

import wandb
wandb.login()

ANSWER_PROMPT_WITH_THEREFORE = "Therefore, the answer is:"
PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_DIR = os.path.join(PROJECT_BASE, "scores", "mmtom_qa_masked")
MMTOM_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(MMTOM_DIR, "saved_outputs")
# openai.api_key = YOUR-OPENAI-KEY

def generate_response_gpt4v(prompt, base64_images, max_tokens = 100, temperature = 0):
    client = OpenAI(api_key='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')
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[0]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[1]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[2]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[3]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[4]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[5]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[6]}",
                    "detail": "low"
                    },
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_images[7]}",
                    "detail": "low"
                    },
                },
            ],
            }
        ],
        max_tokens = max_tokens,
        )
    return response.choices[0].message.content

def read_frame_intervals(parent_path):
    path = parent_path + 'frame_intervals.pik'
    with open(path, 'rb') as f:
        intervals = pickle.load(f)
    return intervals

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError:
        # Create an empty image and return its encoding if the file is not found
        empty_image = Image.new('RGB', (1, 1))
        buffered = io.BytesIO()
        empty_image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def evaluate_answer(response, answer) -> bool:
    if response.startswith("(" + answer + ")") or response.startswith(answer + ")") or response.startswith(answer + ".") or response.startswith(answer + ":") or response.startswith(answer + ",") or "({})".format(answer) in response or answer == response: # a) or a. or a or (a)
        return True
    else:
        return False

def run(args):
    # Load data
    wandb.init(
        project="mmtom_qa",
        name=f"{args.model.split('/')[-1]}_tracing-{args.use_tracing}_cot-{args.use_cot}_{args.run_id}",
        tags=[args.model, args.run_id, f"tracing-{args.use_tracing}", f"cot-{args.use_cot}", args.tracer_type, f"n-{args.n_questions}"],
        save_code=True
    )
    wandb.run.save("../*tracer*.py", base_path="../", policy="now")

    file_path = os.path.join(PROJECT_BASE, "data", "mmtom_qa", "revised_questions.jsonl")
    vqa_dataset = pd.read_json(file_path, lines=True)
    os.makedirs(SCORES_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    model_name = args.model.split("/")[-1] # Get the model name
    file_name = f"{model_name}_tracing-{args.use_tracing}_cot-{args.use_cot}_{args.run_id}.jsonl"

    if args.existing_traces:
        df = pd.read_json(args.existing_traces, lines=True)

    if args.use_tracing:
        model = load_tracer_model(args)
    else:
        model = load_model(args.model, **vars(args))

    # Few shot prompts
    two_shot_prompt = "What's inside the apartment: The apartment consists of a living room, bedroom, kitchen, and bathroom. \nIn the living room, there is a coffee table, a sofa, a cabinet, and a desk. The cabinet houses a dish bowl, a water glass, a bag of chips, and a bottle of wine. \nThe bedroom is furnished with a desk and a coffee table, which has three water glasses on it. \nThe kitchen is equipped with four cabinets, a fridge, a microwave, a kitchen table, and a stove. The first cabinet, from left to right, contains a plate and a bottle of wine. The second cabinet holds a wine glass and a water glass. The third cabinet stores two apples, while the fourth cabinet has a wine glass and an apple. The fridge contains a salmon and an apple. Inside the microwave, there is a condiment bottle and a salmon. The stove houses a cupcake and a plate. \nThe bathroom features a cabinet, which is currently empty. \nActions taken by Mary: Mary is situated in the living room. She proceeds to the kitchen and heads towards the second kitchen cabinet. She opens it, then promptly closes it. Mary then makes her way to the bathroom, approaching the bathroom cabinet. She opens it, then shuts it. She returns to the kitchen, moving towards the microwave, which she opens and then closes. She then opens the fridge, and after a moment, closes it. Mary then walks towards the third kitchen cabinet, opens it, and closes it. She repeats this action with the fourth kitchen cabinet. She then opens the stove, and after a moment, closes it. She walks towards the first kitchen cabinet, opens it, and then closes it. Finally, Mary returns to the living room, preparing to open another cabinet. \nQuestion: If Mary has been trying to get a bag of chips, which one of the following statements is more likely to be true? (a) Mary thinks that the bag of chips is not inside the cabinet. (b) Mary thinks that the bag of chips is inside the cabinet. Please respond with either a or b. \nAnswer: b. \n\n\n\
                    What's inside the apartment: The apartment consists of a bedroom, bathroom, living room, and kitchen. \nIn the bedroom, there is a desk and a coffee table, with a dish bowl placed on the coffee table. \nThe bathroom houses a cabinet, which is currently empty. \nThe living room is furnished with a desk, coffee table, cabinet, and sofa. On the coffee table, there is a remote control, a wine glass, and a dish bowl. The cabinet contains a bag of chips, a water glass, a condiment bottle, and two wine glasses. \nThe kitchen is equipped with four cabinets, a fridge, a stove, a microwave, and a kitchen table. The first kitchen cabinet, from left to right, contains an apple, while the fourth is empty. Inside the fridge, there are two cupcakes, a bottle of wine, three apples, a dish bowl, and a plate. The third kitchen cabinet is empty. The stove houses a salmon. The second kitchen cabinet contains a dish bowl and a wine glass. Lastly, there is a cupcake in the microwave. \nActions taken by Elizabeth: Elizabeth is situated in the kitchen. She strides towards the first kitchen cabinet, opens it, then promptly shuts it. Subsequently, she opens the third kitchen cabinet and closes it as well, before finally making her way towards the fridge. \nQuestion: If Elizabeth doesn't think there is a bottle of wine inside the fridge, which one of the following statements is more likely to be true? (a) Elizabeth has been trying to get a remote control. (b) Elizabeth has been trying to get a bottle of wine. Please respond with either a or b. \nAnswer: a. \n\n\n"
    one_shot_prompt = "What's inside the apartment: The apartment consists of a living room, bedroom, kitchen, and bathroom. \nIn the living room, there is a coffee table, a sofa, a cabinet, and a desk. The cabinet houses a dish bowl, a water glass, a bag of chips, and a bottle of wine. \nThe bedroom is furnished with a desk and a coffee table, which has three water glasses on it. \nThe kitchen is equipped with four cabinets, a fridge, a microwave, a kitchen table, and a stove. The first cabinet, from left to right, contains a plate and a bottle of wine. The second cabinet holds a wine glass and a water glass. The third cabinet stores two apples, while the fourth cabinet has a wine glass and an apple. The fridge contains a salmon and an apple. Inside the microwave, there is a condiment bottle and a salmon. The stove houses a cupcake and a plate. \nThe bathroom features a cabinet, which is currently empty. \nActions taken by Mary: Mary is situated in the living room. She proceeds to the kitchen and heads towards the second kitchen cabinet. She opens it, then promptly closes it. Mary then makes her way to the bathroom, approaching the bathroom cabinet. She opens it, then shuts it. She returns to the kitchen, moving towards the microwave, which she opens and then closes. She then opens the fridge, and after a moment, closes it. Mary then walks towards the third kitchen cabinet, opens it, and closes it. She repeats this action with the fourth kitchen cabinet. She then opens the stove, and after a moment, closes it. She walks towards the first kitchen cabinet, opens it, and then closes it. Finally, Mary returns to the living room, preparing to open another cabinet. \nQuestion: If Mary has been trying to get a bag of chips, which one of the following statements is more likely to be true? (a) Mary thinks that the bag of chips is not inside the cabinet. (b) Mary thinks that the bag of chips is inside the cabinet. Please respond with either a or b. \nAnswer: b. \n\n\n"

    correct = {}
    all = {}
    answers = []
    vqa_subset = vqa_dataset.sample(n=args.n_questions, random_state=123)
    idx = -1
    for _, vqa in tqdm(vqa_subset.iterrows(), total=len(vqa_subset)):

        if args.print:
            print(f"Question type: {vqa['question_type']}")
            print(f"Correct answer: {vqa['answer']}")
        
        idx += 1
        if args.model == 'gpt-4v':
            episode = vqa["episode"]
            end_time = vqa["end_time"]
            interval_path = f'{args.benchmark_video_path}/task_{episode}/'
            video_path = f'{args.benchmark_video_path}/task_{episode}/script/0/'

            # Read frame intervals and extracting end frame time
            intervals = read_frame_intervals(interval_path)
            times = [action[1] for action in intervals]
            end_frame = times[end_time]

            # Construct paths for selected frames and encoding them
            num_frame = 8
            step_size = int(end_frame / (num_frame - 1))
            selected_numbers = [i * step_size for i in range(num_frame)]
            paths = [video_path + f'Action_{selected_number:04d}_0_normal.png' for selected_number in selected_numbers]
            base64_images = [encode_image(path) for path in paths]

        # Construct the question prompt based on the modality and few-shot setting
        question = vqa["question"]
        better_prompt = "Please respond with either (a) or (b) without any explanation."
        question = question.replace("Please respond with either a or b.", better_prompt)

        if args.video_only:
            question = "\nQuestion:" + question.split("\nQuestion:")[-1]
        if args.few_shot == 2:
            input_text = two_shot_prompt + question + "\nAnswer: " # + "Let's think step by step."
        elif args.few_shot == 1:
            input_text = one_shot_prompt + question + "\nAnswer: " # + "Let's think step by step."
        else:
            input_text = question + "\nAnswer: " # + "Let's think step by step."

        correct_answer = vqa['answer']

        if args.existing_traces and idx < len(df):
            df_row = df.iloc[idx]
            input_text = df_row['input_text']
            if args.use_cot:
                simple_question = question.split("\nQuestion:")[-1].replace(better_prompt, "").strip()
                input_text = input_text.replace(ANSWER_PROMPT_WITH_THEREFORE, "")
                input_text = input_text.replace(better_prompt, "")
                input_text = input_text.replace("\nAnswer:", "")
                input_text = input_text.strip() + f"\n\nLet's think step by step."
                cot_reasoning = model.interact(input_text, max_tokens=1024)
                input_text = f"{input_text.strip()}\n\n{cot_reasoning.strip()}\n\n{better_prompt}\n{ANSWER_PROMPT_WITH_THEREFORE}"

            try:
                generated_answer = model.interact(input_text)
            except:
                print("Error in generating answer")
                generated_answer = model.interact(input_text)
        else:
            if args.use_tracing:
                question = "Question:" + question.split("\nQuestion:")[-1]
                target_character = model.identify_target(question)
                generated_thoughts = model.trace(input_text, target_agent=target_character)
                input_text = input_text.replace(better_prompt, "")
                input_text = input_text.replace("\nAnswer:", "").strip()
                input_text = f"{input_text}\n\n{generated_thoughts.strip()}\n\n{better_prompt}\n{ANSWER_PROMPT_WITH_THEREFORE}"

            if args.use_cot:
                input_text = input_text.replace(better_prompt, "")
                input_text = input_text.replace("\nAnswer:", "")
                input_text = input_text.strip() + "\n\nLet's think step by step."
                cot_reasoning = model.interact(input_text, max_tokens=1024, temperature=0.7)
                input_text = f"{input_text}\n\n{cot_reasoning.strip()}\n\n{better_prompt}\n{ANSWER_PROMPT_WITH_THEREFORE}"

            generated_answer = model.interact(input_text)

        if args.print:
            print(Panel(input_text, box=box.SIMPLE_HEAD, title="Input Text", style="cyan", expand=False))
            print(Panel(question, box=box.SIMPLE_HEAD, title="Question", expand=False))
            print(Panel(vqa['answer'], box=box.SIMPLE_HEAD, title="Correct Answer", expand=False))

        if args.model == "deepseek-ai/DeepSeek-R1":
            thoughts = generated_answer.split("</think>")[0].strip()
            prediction = generated_answer.split("</think>")[-1].strip()
        elif args.model.startswith("o1") or args.model.startswith("o3"):
            prediction = generated_answer['response']
            n_reasoning_tokens = generated_answer['n_reasoning_tokens']
        else:
            prediction = generated_answer

        # Evaluate the correctness
        _evaluation_result = evaluate_answer(prediction, correct_answer)
        if _evaluation_result:
            correct[f"{str(vqa['test'])} + type {vqa['question_type']}"] = correct.get(f"{str(vqa['test'])} + type {vqa['question_type']}", 0) + 1
            answers.append(1)
            if args.print:
                print(Panel(f"{generated_answer} Question_type: {vqa['question_type']}", title="Generated Answer", style="blue", expand=False))
        else:
            answers.append(0)
            if args.print:
                print(Panel(f"{generated_answer} Question_type: {vqa['question_type']}", title="Generated Answer", style="red", expand=False))
        all[f"{str(vqa['test'])} + type {vqa['question_type']}"] = all.get(f"{str(vqa['test'])} + type {vqa['question_type']}", 0) + 1

        output_save_loc = os.path.join(OUTPUTS_DIR, file_name)

        with open(output_save_loc, 'a') as f:
            if args.model == "deepseek-ai/DeepSeek-R1":
                f.write(json.dumps({
                    'input_text': input_text,
                    'generated_answer': generated_answer,
                    'correct_answer': correct_answer,
                    'question_type': vqa['question_type'],
                    'test': vqa['test'],
                    'thoughts': thoughts,
                    'prediction': prediction,
                    'eval_result': _evaluation_result,
                }) + "\n")
            elif args.model.startswith("o1") or args.model.startswith("o3"):
                f.write(json.dumps({
                    'input_text': input_text,
                    'generated_answer': generated_answer,
                    'correct_answer': correct_answer,
                    'question_type': vqa['question_type'],
                    'test': vqa['test'],
                    'n_reasoning_tokens': n_reasoning_tokens,
                    'eval_result': _evaluation_result,
                }) + "\n")
            else:
                f.write(json.dumps({
                    'input_text': input_text,
                    'generated_answer': generated_answer,
                    'correct_answer': correct_answer,
                    'question_type': vqa['question_type'],
                    'test': vqa['test'],
                    'eval_result': _evaluation_result,
                }) + "\n")
            

    # Show results
    tests = [['filtered_graph', 'filtered_actions'], ['filtered_init_graph', 'actions'], ['graph', 'filtered_actions'], ['init_graph', 'actions']]
    question_types = [1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 2.4]
    # for test in tests:
    #     for question_type in question_types:
    #         index = f"{str(test)} + type {question_type}"
    #         print(f"{index}: {correct.get(index, 0)} out of {all.get(index, 0)} questions are answered correctly.")
    
    # Create a table to show results
    results = {
        '': [f"type {question_type}" for question_type in question_types],
        # str(tests[0]): [f"{correct.get(f'{str(tests[0])} + type {question_type}', 0)}/{all.get(f'{str(tests[0])} + type {question_type}', 0)}" for question_type in question_types],
        # str(tests[1]): [f"{correct.get(f'{str(tests[1])} + type {question_type}', 0)}/{all.get(f'{str(tests[1])} + type {question_type}', 0)}" for question_type in question_types],
        # str(tests[2]): [f"{correct.get(f'{str(tests[2])} + type {question_type}', 0)}/{all.get(f'{str(tests[2])} + type {question_type}', 0)}" for question_type in question_types],
        str(tests[3]): [f"{correct.get(f'{str(tests[3])} + type {question_type}', 0)}/{all.get(f'{str(tests[3])} + type {question_type}', 0)}" for question_type in question_types],
        "avg": [round(correct.get(f'{str(tests[3])} + type {question_type}', 0) / all.get(f'{str(tests[3])} + type {question_type}', 0), 3)*100 for question_type in question_types],
    }
    df = pd.DataFrame(results)

    print(df)
    df.columns = ['q_type', 'init_graph-actions', 'avg']
    df['macro_q_type'] = df['q_type'].apply(lambda x: x.split('.')[0])
    df['correct'] = df['init_graph-actions'].apply(lambda x: int(x.split("/")[0]))
    df['count'] = df['init_graph-actions'].apply(lambda x: int(x.split("/")[1]))
    macro_result = (df.groupby('macro_q_type')['correct'].sum() / df.groupby('macro_q_type')['count'].sum()).to_dict()
    df.drop(columns=['correct', 'count', 'macro_q_type'], inplace=True)
    df.set_index('q_type', inplace=True)
    total_n_questions = sum(all.values())
    total_correct = sum(correct.values())
    total_avg = total_correct / total_n_questions
    rounded_avg = round(total_avg, 2)
    print(Panel(f"Total: {total_correct} out of {total_n_questions} questions are answered correctly. {rounded_avg}", box=box.SIMPLE_HEAD, title="Results", style="magenta", expand=False))
    result_dict = df.to_dict()
    result_dict['total_avg'] = rounded_avg
    result_dict = {**result_dict, **macro_result}
    model_name = args.model.split("/")[-1] # Get the model name
    file_name = f"{model_name}_tracing-{args.use_tracing}_cot-{args.use_cot}_{args.run_id}.json"
    with open(os.path.join(SCORES_DIR, file_name), 'w') as f:
        json.dump(result_dict, f, indent=4)

    wandb.log(result_dict)
    wandb.finish()
        
if __name__ == "__main__":
    parser = get_tracer_parser()
    parser.add_argument("--few_shot", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--video_only", action='store_true')
    parser.add_argument('--model',
                        type=str,
                        help='name of the model to run evaluation',
    )
    parser.add_argument('--batch-size',
                        type=int,
                        default=50,
                        help='batch size for evaluation',
    )
    parser.add_argument('--use-cot',
                        type=bool,
                        default=False,
                        help='whether to use cot or not',
    )
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--run-id', type=str, required=True, help='Run ID')
    parser.add_argument('--print', action='store_true', help='whether to print the outputs')
    parser.add_argument('--n-questions', type=int, default=500, help='number of questions to evaluate')
    parser.add_argument('--reasoning-effort', type=str, help='Reasoning effort')
    args = parser.parse_args()
    run(args)
