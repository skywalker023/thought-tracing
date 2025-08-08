import os
import sys
import csv
import json
import random
from tqdm import tqdm

sys.path.append("..")
from agents.gpt import ConversationalGPTBaseAgent
from agents.load_model import load_model
from tracer import load_tracer_model, get_tracer_parser
from utils import log_outputs, read_logs

import colorful as cf
cf.use_true_colors()
cf.use_style('monokai')

from rich import print
from rich.panel import Panel
from rich import box
import pandas as pd

import wandb
wandb.login()

PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIGTOM_DIR = os.path.join(PROJECT_BASE, "data", "bigtom")
random.seed(123)
ANSWER_PROMPT_WITH_THEREFORE = "Therefore, the answer is:"

def load_bigtom(init_belief='0_backward', variable='belief', condition='false_belief', offset=0, num_probs=200, mcq=True):
    bigtom_data = []
    csv_name = os.path.join(BIGTOM_DIR, f'{init_belief}_{variable}_{condition}', 'stories.csv')
    with open(csv_name, "r") as f:
        reader = csv.reader(f, delimiter=";")
        condition_rows = list(reader)

    for row in tqdm(condition_rows[offset:offset+num_probs]):
        story = row[0]
        question_orig = row[1]
        question = row[1]
        true_answer, wrong_answer = row[2], row[3]
        answers = [true_answer, wrong_answer]
        random.shuffle(answers)
        if answers[0] == true_answer:
            answer_key = 'a)'
            negative_answer_key = 'b)'
            true_answer = 'a) ' + true_answer
            wrong_answer = 'b) ' + wrong_answer
        else:
            answer_key = 'b)'
            negative_answer_key = 'a)'
            true_answer = 'b) ' + true_answer
            wrong_answer = 'a) ' + wrong_answer
        if mcq:
            question = f"{question}\nChoose one of the following:\na){answers[0]}\nb){answers[1]}"
        bigtom_data.append({'context': story, 'question': question, 'true_answer': true_answer, 'wrong_answer': wrong_answer, 'true_answer_key': answer_key, 'wrong_answer_key': negative_answer_key, 'original_question': question_orig})

    return bigtom_data

def grade_answer_with_model(query, predicted_answer, true_answer, wrong_answer):
    grade_file = os.path.join(BIGTOM_DIR, "grade.txt")
    with open(grade_file, 'r') as f:
        instruction = f.read()

    instruction_prompt = """{instruction}

Here is the question:
{query}
Here is the true answer:
{true_answer}
Here is the false answer:
{wrong_answer}
Here is the predicted answer:
{predicted_answer}
Is the predicted answer close to the true answer compared to the false answer? Answer True or False.
Answer:"""
    prompt = instruction_prompt.format(instruction=instruction, query=query, true_answer=true_answer, wrong_answer=wrong_answer, predicted_answer=predicted_answer)
    model = 'gpt-3.5-turbo-0125'
    gpt = ConversationalGPTBaseAgent({'model': model, 'temperature': 0, 'top_p': 1.0, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'max_tokens': 16})
    results = gpt.interact(prompt)
    return results[0]

def parse_chat_response(response):
    # """BigToM's original parser function"""
    # answer_idx = response.find('Answer:')
    # return response[answer_idx+8:].strip()

    # my own
    if isinstance(response, str) and len(response) == 1:
        return response + ")"
    else:
        return response

def grade_bigtom(question_set, predicted_answer):
    answer_key = question_set['true_answer_key']
    negative_answer_key = question_set['wrong_answer_key']
    true_answer = question_set['true_answer']
    wrong_answer = question_set['wrong_answer']
    question_orig = question_set['original_question']

    if answer_key in predicted_answer.lower():
        graded_answer = 'True'
    elif negative_answer_key in predicted_answer.lower():
        graded_answer = 'False'
    else:
        print(f"predicted answer: {predicted_answer}")
        print(f"true answer: {true_answer}")
        print(f"wrong answer: {wrong_answer}")
        graded_answer = grade_answer_with_model(question_orig, predicted_answer, true_answer, wrong_answer).strip()
        print(f"graded answer: {graded_answer}")

    if graded_answer is None:
        return False
    if graded_answer.lower() == 'true':
        return True
    elif graded_answer.lower() == 'false':
        return False
    else:
        return False

class TomTester():
    def __init__(self, args):
        self.args = args
        if args.use_tracing:
            self.agent = load_tracer_model(args)
        else:
            self.agent = load_model(args.model, **vars(args))
            self.args.tracing_model = "none"
            self.args.n_hypotheses = 0
        self.data = load_bigtom(init_belief=args.init_belief, variable=args.variable, condition=args.condition, num_probs=args.num_probs, mcq=True)

    def run(self):
        wandb.init(
            project="bigtom",
            name=f"{self.args.model.split('/')[-1]}_tracing-{self.args.use_tracing}_cot-{self.args.use_cot}_{self.args.run_id}",
            tags=[self.args.model, self.args.run_id, f"tracing-{self.args.use_tracing}", f"cot-{self.args.use_cot}", f"init_belief-{self.args.init_belief}", f"variable-{self.args.variable}", f"condition-{self.args.condition}", f"debug-{self.args.debug}", self.args.tracer_type, f"num-probs-{self.args.num_probs}"],
            save_code=True
        )
        wandb.run.save("../*tracer*.py", base_path="../", policy="now")
        eval_instruction = "Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'."
        predicted_answers = []
        graded_results = []

        logs = read_logs("bigtom_" + self.args.run_id, self.args.model, self.args.tracing_model, str(self.args.n_hypotheses), condition=self.args.condition)
        if logs is not None:
            predicted_answers = logs['predicted_answer'].tolist()
            graded_results = logs['graded_result'].tolist()
            last_save_idx = logs.index[-1]
        else:
            last_save_idx = -1

        # target_data = random.sample(self.data, k=self.args.num_probs)
        target_data = self.data

        for idx, data in tqdm(enumerate(target_data), total=len(target_data)):
            if idx <= last_save_idx:
                continue

            eval_template = """{instruction}

Story: {story}
Question: {question}
Answer:"""
            eval_input_prompt = eval_template.format(instruction=eval_instruction, story=data['context'], question=data['question'])

            # Trace the thought
            if self.args.use_tracing:
                thought = self.agent.trace(eval_input_prompt)
                prompt_for_final_answer = eval_input_prompt.removesuffix("\nAnswer:") + f"\n\n{thought}\n{ANSWER_PROMPT_WITH_THEREFORE}"
                if "qwen" in self.args.model.lower():
                    response, logprobs = self.agent.base_model.interact(prompt_for_final_answer, temperature=0, max_tokens=128)
                else:
                    response = self.agent.base_model.interact(prompt_for_final_answer, temperature=0, max_tokens=128)
                predicted_answer = parse_chat_response(response)
                graded_result = grade_bigtom(data, predicted_answer)
                predicted_answers.append(predicted_answer)
                graded_results.append(graded_result)
            else:
                if self.args.use_cot:
                    response = self.agent.cot(eval_input_prompt, temperature=0, max_tokens=256)
                else:
                    if "qwen" in self.args.model.lower():
                        response, logprobs = self.agent.interact(eval_input_prompt, temperature=0, max_tokens=128)
                    else:
                        response = self.agent.interact(eval_input_prompt, temperature=0, max_tokens=128)
                predicted_answer = parse_chat_response(response)
                graded_result = grade_bigtom(data, predicted_answer)
                predicted_answers.append(predicted_answer)
                graded_results.append(graded_result)
                thought = "No tracing."

            if self.args.print:
                print(Panel(f"idx: {idx}\n{eval_input_prompt}", title="Input", box=box.SIMPLE_HEAD, style="green"))
                if graded_result is False:
                    print(cf.bold | cf.ghostWhite(">>> Entire Context: " + data['context']))
                    print(cf.bold | cf.ghostWhite(">>> Question: " + data['original_question']))
                    print()
                    if self.args.use_tracing:
                        print(Panel(f">>> Incorrect predicted answer: {predicted_answer}", title="Output", box=box.SIMPLE_HEAD, style="red"))
                else:
                    if self.args.use_tracing:
                        print(Panel(f">>> Correct traced thought: {thought}", title="Thought", box=box.SIMPLE_HEAD, style="blue"))

                if not graded_result:
                    print(Panel(f">>> Wrong predicted answer: {predicted_answer}", title="Output", box=box.SIMPLE_HEAD, style="red"))


            print(">>> True answer: ", data['true_answer'])
            print(Panel(">>> Cumulative accuracy: " + str(sum(graded_results) / len(graded_results)), title="Accuracy", box=box.SIMPLE_HEAD, style="blue"))
            # print(">>> Total examples: ", len(graded_results))
            print()
            print("=====================================================================================================")
            print()

            log_outputs([{'context': data['context'], 'question': data['question'], 'true_answer': data['true_answer'], 'predicted_answer': predicted_answer, 'graded_result': graded_result, 'thought_traces': thought}], model=self.args.model, dataset="bigtom_" + self.args.run_id, tracing_model=self.args.tracing_model, particle=str(self.args.n_hypotheses), condition=self.args.condition)
        
        SCORES_DIR = os.path.join(PROJECT_BASE, "scores", "bigtom", args.init_belief + "_" + args.variable, args.run_id)
        os.makedirs(SCORES_DIR, exist_ok=True)
        if self.args.use_tracing:
            tracer_name = self.args.tracing_model.replace("/", "-")
            file_id = f"tracer-{tracer_name}_runid-{self.args.run_id}"
        else:
            base_name = args.model.replace("/", "-")
            file_id = f"model-{base_name}_runid-{self.args.run_id}"
            if self.args.use_cot:
                file_id += "_cot"
        score_file_name = os.path.join(SCORES_DIR, f"{file_id}.json")
        report = {'accuracy': sum(graded_results) / len(graded_results), 'graded_results': graded_results, 'predicted_answers': predicted_answers}
        with open(score_file_name, 'w') as f:
            json.dump(report, f, indent=4)
        
        wandb_loggings = {'accuracy': report['accuracy']}
        wandb.log(wandb_loggings)
        wandb.finish()

class CustomBigTomTester(TomTester):
    def __init__(self, args):
        self.args = args
        if args.use_tracing:
            self.agent = load_tracer_model(args)
        else:
            self.agent = load_model(args.model, **vars(args))
            self.args.tracing_model = "none"
            self.args.n_hypotheses = 0
        bigtom_file = os.path.join(BIGTOM_DIR, f"bigtom_{args.target_subset}.json")
        self.data = pd.read_json(bigtom_file).T.to_dict(orient='records')

    def run(self):
        wandb.init(
            project="bigtom600",
            name=f"{self.args.model.split('/')[-1]}_tracing-{self.args.use_tracing}_cot-{self.args.use_cot}_{self.args.run_id}",
            tags=[self.args.model, self.args.run_id, f"tracing-{self.args.use_tracing}", f"cot-{self.args.use_cot}", self.args.target_subset, self.args.tracer_type],
            save_code=True
        )
        wandb.run.save("../*tracer*.py", base_path="../", policy="now")
        eval_instruction = "Answer the questions based on the context. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'."
        predicted_answers = []
        graded_results = []

        # eval_instruction_tracing = "Answer the questions based on the context and the given thoughts. Keep your answer concise, few words are enough, maximum one sentence. Answer as 'Answer:<option>)<answer>'."
        # predicted_answers_tracing = []
        # graded_results_tracing = []

        logs = read_logs("bigtom_" + self.args.run_id, self.args.model, self.args.tracing_model, str(self.args.n_hypotheses), condition=self.args.condition)
        if logs is not None:
            predicted_answers = logs['predicted_answer'].tolist()
            graded_results = logs['graded_result'].tolist()
            last_save_idx = logs.index[-1]
        else:
            last_save_idx = -1

        target_data = self.data
        if self.args.existing_traces:
            _existing_traces = pd.read_json(self.args.existing_traces, lines=True)
            existing_traces = _existing_traces['thought_traces'].tolist()

        for idx, data in tqdm(enumerate(target_data), total=len(target_data)):
            if idx <= last_save_idx:
                continue

            eval_template = """{instruction}

Story: {story}
Question: {question}
Answer:"""
            eval_input_prompt = eval_template.format(instruction=eval_instruction, story=data['context'], question=data['question'])

            # Trace the thought
            if self.args.use_tracing:
                if self.args.existing_traces:
                    thought = existing_traces[idx] # get trace from the existing traces
                else:
                    thought = self.agent.trace(eval_input_prompt)
                
                if self.args.use_cot:
                    # prompt_for_cot = eval_input_prompt.removesuffix("\nAnswer:") + f"\n\n{thought}\n\nQuestion: {data['question']}\n\nNow let's think step by step."
                    prompt_for_cot = eval_input_prompt.removesuffix("\nAnswer:").split("Answer as 'Answer:<option>)<answer>'.")[-1].strip() + f"\n\n{thought}\n\nNow let's think step by step based on the above."
                    cot_response = self.agent.base_model.interact(prompt_for_cot, temperature=0, max_tokens=512)
                    prompt_for_final_answer = f"{prompt_for_cot}\n{cot_response}\n\n{ANSWER_PROMPT_WITH_THEREFORE}"
                else:
                    prompt_for_final_answer = eval_input_prompt.removesuffix("\nAnswer:") + f"\n\n{thought}\n{ANSWER_PROMPT_WITH_THEREFORE}"

                if "qwen" in self.args.model.lower():
                    response, logprobs = self.agent.base_model.interact(prompt_for_final_answer, temperature=0, max_tokens=128)
                else:
                    response = self.agent.base_model.interact(prompt_for_final_answer, temperature=0, max_tokens=128)
                predicted_answer = parse_chat_response(response)
                prediction = response
                graded_result = grade_bigtom(data, predicted_answer)
                predicted_answers.append(predicted_answer)
                graded_results.append(graded_result)
            else:
                if self.args.use_cot:
                    eval_input_prompt = eval_input_prompt.removesuffix("\nAnswer:").split("Answer as 'Answer:<option>)<answer>'.")[-1].strip()
                    cot_response = self.agent.cot(eval_input_prompt, temperature=0, max_tokens=512)
                    eval_input_prompt = f"{eval_input_prompt}\n\n{cot_response}\n\n{ANSWER_PROMPT_WITH_THEREFORE}"
                response = self.agent.interact(eval_input_prompt, temperature=0, max_tokens=128)

                if self.args.model == "deepseek-ai/DeepSeek-R1":
                    thought = response.split("</think>")[0].strip()
                    prediction = response.split("</think>")[-1].strip()
                elif self.args.model.startswith("o1") or self.args.model.startswith("o3"):
                    prediction = response['response']
                    n_reasoning_tokens = response['n_reasoning_tokens']
                else:
                    prediction = response

                predicted_answer = parse_chat_response(prediction)
                graded_result = grade_bigtom(data, predicted_answer)
                predicted_answers.append(predicted_answer)
                graded_results.append(graded_result)
                if not self.args.model == "deepseek-ai/DeepSeek-R1":
                    thought = "No tracing."

            if self.args.print:
                print(Panel(f"idx: {idx}\n{eval_input_prompt}", title="Input", box=box.SIMPLE_HEAD, style="green"))
                if graded_result is False:
                    print(cf.bold | cf.ghostWhite(">>> Entire Context: " + data['context']))
                    print(cf.bold | cf.ghostWhite(">>> Question: " + data['original_question']))
                    print()
                    # if self.args.use_tracing:
                    print(Panel(f">>> Incorrect predicted answer: {response}", title="Output", box=box.SIMPLE_HEAD, style="red"))
                else:
                    if self.args.use_tracing:
                        print(Panel(f">>> Correct traced thought: {thought}", title="Thought", box=box.SIMPLE_HEAD, style="blue"))
                    else:
                        print(Panel(f">>> Correct predicted answer: {response}", title="Output", box=box.SIMPLE_HEAD, style="blue"))
                    if self.args.use_cot:
                        print(Panel(f">>> COT response: {cot_response}", title="COT Response", box=box.SIMPLE_HEAD, style="blue"))

                if not graded_result:
                    if self.args.use_tracing:
                        print(Panel(f">>> Incorrect traced thought: {thought}", title="Thought", box=box.SIMPLE_HEAD, style="red"))
                    if self.args.use_cot:
                        print(Panel(f">>> COT response: {cot_response}", title="COT Response", box=box.SIMPLE_HEAD, style="red"))
                    print(Panel(f">>> Wrong predicted answer: {response}", title="Output", box=box.SIMPLE_HEAD, style="red"))


            print(">>> True answer: ", data['true_answer'])
            print(Panel(">>> Cumulative accuracy: " + str(sum(graded_results) / len(graded_results)), title="Accuracy", box=box.SIMPLE_HEAD, style="blue"))
            # print(">>> Total examples: ", len(graded_results))
            print()
            print("=====================================================================================================")
            print()

            if self.args.model.startswith("o1") or self.args.model.startswith("o3"):
                log_outputs([{'context': data['context'], 'question': data['question'], 'true_answer': data['true_answer'], 'predicted_answer': predicted_answer, 'raw_prediction': prediction, 'graded_result': graded_result, 'thought_traces': thought, 'n_reasoning_tokens': n_reasoning_tokens}], model=self.args.model, dataset="bigtom_" + self.args.run_id, tracing_model=self.args.tracing_model, particle=str(self.args.n_hypotheses), condition=self.args.condition)
            else:
                log_outputs([{'context': data['context'], 'question': data['question'], 'true_answer': data['true_answer'], 'predicted_answer': predicted_answer, 'raw_prediction': prediction, 'graded_result': graded_result, 'thought_traces': thought}], model=self.args.model, dataset="bigtom_" + self.args.run_id, tracing_model=self.args.tracing_model, particle=str(self.args.n_hypotheses), condition=self.args.condition)
        
        SCORES_DIR = os.path.join(PROJECT_BASE, "scores", "bigtom", self.args.target_subset, self.args.run_id)
        os.makedirs(SCORES_DIR, exist_ok=True)
        if self.args.use_tracing:
            tracer_name = self.args.tracing_model.replace("/", "-")
            file_id = f"tracer-{tracer_name}_runid-{self.args.run_id}"
        else:
            base_name = args.model.replace("/", "-")
            file_id = f"model-{base_name}_runid-{self.args.run_id}"
            if self.args.use_cot:
                file_id += "_cot"
        score_file_name = os.path.join(SCORES_DIR, f"{file_id}.json")
        report = {'accuracy': sum(graded_results) / len(graded_results), 'graded_results': graded_results, 'predicted_answers': predicted_answers}

        evaluated_data = pd.DataFrame(target_data)
        evaluated_data['graded_results'] = graded_results
        # group by the task_type and get the average accuracy
        task_accuracy = evaluated_data.groupby('task_type')['graded_results'].mean().to_dict()

        with open(score_file_name, 'w') as f:
            json.dump(report, f, indent=4)
        
        wandb_loggings = {'accuracy': report['accuracy'], **task_accuracy}
        wandb.log(wandb_loggings)
        wandb.finish()

def main(args):
    if args.target_subset is None:
        tester = TomTester(args)
    elif args.target_subset in ['agree90', 'agree80', 'agree70', 'agree50', 'confusing']:
        tester = CustomBigTomTester(args)
    tester.run()

if __name__ == "__main__":
    # argparse
    parser = get_tracer_parser()
    parser.add_argument('--use-cot',
                        default=False,
                        help='whether to run the model with zero-shot cot',
    )
    parser.add_argument('--init-belief',
                        type=str,
                        default='0_backward',
                        help='mentions initial belief state',
                        )
    parser.add_argument('--variable',
                        type=str,
                        default='belief',
                        help='variable to test',
                        )
    parser.add_argument('--condition',
                        type=str,
                        default='false_belief',
                        help='condition to test',
                        )
    parser.add_argument('--num-probs',
                        type=int,
                        default=200,
                        help='number of problems to test. Maximum is 200',
                        )
    parser.add_argument('--run-id',
                        type=str,
                        required=True,
                        help='run id',
                        )
    parser.add_argument('--model',
                        type=str,
                        # 'meta-llama/Llama-2-13b-chat-hf', 'gpt-4-0125-preview', 'lmsys/vicuna-13b-v1.5'
                        help='name of the model to evaluate',
    )
    parser.add_argument('--output-dir',
                        type=str,
                        default='outputs',
                        help='directory to save the outputs',)
    parser.add_argument('--print',
                        type=bool,
                        default=False,
                        help='whether to print the outputs',
    )
    parser.add_argument('--target-subset',
                        type=str,
                        default=None,
                        choices=['agree90', 'agree80', 'agree70', 'agree50', 'confusing'],
                        help='target subset to evaluate',
    )
    parser.add_argument('--reasoning-effort', type=str, help='Reasoning effort')
    args = parser.parse_args()
    main(args)
