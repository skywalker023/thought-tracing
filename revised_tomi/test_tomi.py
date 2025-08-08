import os
import sys
import json
import argparse
import pandas as pd
from collections import Counter
from tqdm import tqdm
import colorful as cf
cf.use_true_colors()
cf.use_style('monokai')

from rich import print
from rich.panel import Panel
from rich import box

sys.path.append("..")
from agents.load_model import load_model
from tracer import load_tracer_model, get_tracer_parser

import wandb
wandb.login()

PROJECT_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOMI_QUESTION_TYPES = [
    'reality', 'memory',
    'first_order_0_tom', 'first_order_1_tom', 'first_order_0_no_tom', 'first_order_1_no_tom',
    'second_order_0_tom', 'second_order_1_tom', 'second_order_0_no_tom', 'second_order_1_no_tom'
]

ANSWER_PROMPT_WITH_THEREFORE = "Therefore, the short one-sentence answer specifying the most detailed location including both the container and the place (e.g., from A in B) without any explanation is:" # one-sentence
ANSWER_PROMPT = "The short one-sentence answer specifying the most detailed location including both the container and the place (e.g., from A in B) without any explanation is:" # one-sentence

def loadFileWithoutMetadata(fn, tomi_set, story_number_limit=-1):
    TOMI_DIR = os.path.join(PROJECT_BASE, "data", tomi_set)
    data = []
    d = {"story": [], "cands": []}

    file_location = os.path.join(TOMI_DIR, f"{fn}.txt")
    for l in open(file_location, "r"):
        if "\t" in l:
            q, a, i = l.strip().split("\t")
            d["question"] = q.split(" ", 1)[1]
            d["answer"] = a.replace("_", " ")
            d["i"] = int(i)
            data.append(d)

            if len(data) >= story_number_limit > 0:
                break

            d = {"story": [], "cands": []}
        else:
            sent = l.strip().split(" ", 1)[1]
            if not sent.endswith("."):
                sent += "."
            cand = sent.strip(".").rsplit(" ")[-1]
            d["cands"].append(cand)
            d["story"].append(sent)

    df = pd.DataFrame(data)
    df["story"] = df["story"].apply(" ".join).str.replace("_", " ")
    df["cands"] = df["cands"].apply(lambda x: list({c.replace("_", " ") for c in x}))
    return df


def loadFileWithCleanQuestionsAndQuestionTypes(fn, tomi_set, story_number_limit=-1):
    TOMI_DIR = os.path.join(PROJECT_BASE, "data", tomi_set)
    df1 = loadFileWithoutMetadata(fn, tomi_set, story_number_limit=story_number_limit)
    df1['question'] = df1['question'].apply(lambda x: x.replace('_', ' ').replace('-', ' - '))
    print(len(df1['question']))

    question_type_file = os.path.join(TOMI_DIR, f"{fn}.trace")
    if os.path.exists(question_type_file):
        with open(question_type_file, 'r') as f:
            df1['qTypeRaw'] = [line.strip().split(',')[-2] for line in f.readlines()]
    else:
        print(f"{question_type_file} not found, assigning same type to all questions.")
        df1['qTypeRaw'] = ['first_order_0_tom' for _ in range(len(df1['question']))]

    return df1

def main(args):
    TOMI_DIR = os.path.join(PROJECT_BASE, "revised_tomi", args.tomi_set)
    SAVED_INPUTS = os.path.join(TOMI_DIR, "saved_inputs")
    os.makedirs(SAVED_INPUTS, exist_ok=True)
    if args.use_tracing:
        tracer_name = args.tracing_model.replace("/", "-")
        file_id = f"tracer-{tracer_name}_runid-{args.run_id}_use_cot-{args.use_cot}_num-qtypes-{str(args.max_questions_per_type)}"
    else:
        base_name = args.model.replace("/", "-")
        file_id = f"model-{base_name}_runid-{args.run_id}_use_cot-{args.use_cot}_num-qtypes-{str(args.max_questions_per_type)}"

    df = loadFileWithCleanQuestionsAndQuestionTypes(args.input_file, args.tomi_set)
    df = df.sample(n=2000, random_state=123)
    args.answer_prompt = ANSWER_PROMPT
    args.answer_prompt_with_therefore = ANSWER_PROMPT_WITH_THEREFORE

    correct_per_question_type = Counter()
    cumulative_results = []

    # if args.use_tracing:
    if args.existing_traces:
        saved_input_df = pd.read_json(args.existing_traces, lines=True)
        last_idx = saved_input_df.index[-1]

    if args.existing_savepoint:
        saved_input_df = pd.read_json(args.existing_savepoint, lines=True)
        old_batch_size = len(saved_input_df.iloc[-1]['input_batch'])
        assert old_batch_size == args.batch_size, f"Batch size mismatch: {old_batch_size} vs {args.batch_size}"
        last_idx = (saved_input_df.index[-1]) * old_batch_size
        # load the answers, update correct_per_question_type, and cumulative_results
        for i, row in saved_input_df.iterrows():
            for j, answer in enumerate(row['answer']):
                if args.model == "deepseek-ai/DeepSeek-R1":
                    prediction = row['predictions'][j].split("</think>")[-1].strip()
                else:
                    prediction = row['predictions'][j]
                # if answer['answer'].lower() in row['predictions'][j].lower():
                if answer['answer'].lower() in prediction.lower():
                    correct_per_question_type[answer['qTypeRaw']] += 1
                    cumulative_results.append(True)
                else:
                    cumulative_results.append(False)
    else:
        last_idx = -1

    if args.use_tracing:
        agent = load_tracer_model(args=args)
    else:
        agent = load_model(model_name=args.model, **vars(args))
    
    remaining_questions_by_type = {
        question_type: args.max_questions_per_type * (1 if 'tom' in question_type else 2)
        for question_type in TOMI_QUESTION_TYPES
    }

    wandb.init(
        project=f"{args.tomi_set}_lower",
        name=f"{args.model.split('/')[-1]}_tracing-{args.use_tracing}_cot-{args.use_cot}_{args.run_id}",
        tags=[args.model, args.run_id, f"tracing-{args.use_tracing}", f"cot-{args.use_cot}", args.tracer_type],
        save_code=True
    )
    wandb.run.save("../*tracer*.py", base_path="../", policy="now")

    input_prompt_list = []
    answer_list = []
    total_per_question_type = Counter()
    for i, row in df.iterrows():
        # if i % args.max_questions_per_type == 0:
        #     print('STORY #', i)

        if remaining_questions_by_type.get(row['qTypeRaw'], 0) == 0:
            continue
        remaining_questions_by_type[row['qTypeRaw']] -= 1
        total_per_question_type[row['qTypeRaw']] += 1

        # Baseline runs
        reconstructed_story = row['story'].strip('.')
        main_question = row['question']

        # Build prompts
        prompt = f"{reconstructed_story}.\n\nQuestion: {main_question}\nAnswer:"

        input_prompt_list.append(prompt)
        answer_list.append({'qTypeRaw': row['qTypeRaw'], 'answer': row['answer']})
    
    for i, (prompt, answer) in enumerate(tqdm(zip(input_prompt_list, answer_list), total=len(input_prompt_list))):
        if args.existing_savepoint and i <= last_idx:
            continue
        
        # make batch according to batch size
        if i % args.batch_size == 0:
            if i + args.batch_size < len(input_prompt_list):
                batch = input_prompt_list[i:i + args.batch_size]
                answers = answer_list[i:i + args.batch_size]
            else:
                batch = input_prompt_list[i:]
                answers = answer_list[i:]

            if args.use_tracing and args.existing_traces and i <= last_idx:
                input_batch = saved_input_df.iloc[i]['input_batch']
            elif args.use_tracing:
                assert args.batch_size == 1, "Tracing only works with batch size 1"
                use_tracings = [False if a['qTypeRaw'] in ['reality', 'memory'] else True for a in answers]
                generated_thoughts = agent.batch_trace(batch, use_tracings)
                if generated_thoughts[0] != '':
                    input_batch = [b.removesuffix("\nAnswer:") + f"\n\n{thought.strip()}\n\n{ANSWER_PROMPT_WITH_THEREFORE}" if thought != '' else b for b, thought in zip(batch, generated_thoughts)]
                else:
                    input_batch = batch
            else:
                input_batch = batch

            if args.use_cot:
                questions = [b.split("Question: ")[-1].replace("\nAnswer:", "") for b in batch]
                input_batch = [b.replace("Answer:", "").replace(ANSWER_PROMPT_WITH_THEREFORE, "").strip() for b in input_batch]
                if args.use_tracing:
                    input_batch = [b + f"\n\nQuestion: {q}\nLet's think step by step.\n" for b, q in zip(input_batch, questions)]
                else:
                    input_batch = [b + "\nLet's think step by step.\n" for b in input_batch]
                cot_responses = agent.batch_interact(input_batch, temperature=0, max_tokens=512)
                input_batch = [b.strip() + f"\n\n{cot_response}\n\n{ANSWER_PROMPT_WITH_THEREFORE}" for b, cot_response in zip(input_batch, cot_responses)]
            else:
                input_batch = [b.replace("Answer:", ANSWER_PROMPT) for b in input_batch]

            if args.model.startswith("o1") or args.model.startswith("o3"):
                generated_output = agent.batch_interact(input_batch, temperature=0)
                generated_answers = [output['response'] for output in generated_output]
                n_reasoning_tokens = [output['n_reasoning_tokens'] for output in generated_output]
            else:
                generated_answers = agent.batch_interact(input_batch, temperature=0, max_tokens=256)

            results = []

            for idx, generated_answer in enumerate(generated_answers):
                if args.model == "deepseek-ai/DeepSeek-R1": 
                    thoughts = generated_answer.split("</think>")[0].strip()
                    prediction = generated_answer.split("</think>")[-1].strip()
                else:
                    prediction = generated_answer
                result = answers[idx]['answer'].lower() in prediction.lower()
                correct_per_question_type[answers[idx]['qTypeRaw']] += result
                results.append(result)
                if args.print:
                    print(Panel(input_batch[0], title="Input", box=box.SIMPLE_HEAD))
                    print(Panel(f"Answer: {answers[idx]['answer']}", title="Correct Answer"))
                    if result:
                        print(Panel(generated_answer, title="Generated Answer", style="blue"))
                    else:
                        print(Panel(generated_answer, title="Generated Answer", style="red"))
                cumulative_results.append(result)
                print(cf.bold('Cumulative Results: ' + str(sum(cumulative_results)) + "/" + str(len(cumulative_results))))

            with open(os.path.join(SAVED_INPUTS, f"{file_id}.jsonl"), 'a') as f:
                if args.model.startswith("o1") or args.model.startswith("o3"):
                    f.write(json.dumps({'input_batch': input_batch, 'answer': answers, 'predictions': generated_answers, 'n_reasoning_tokens': n_reasoning_tokens}) + "\n")
                elif args.use_tracing:
                    f.write(json.dumps({'input_batch': input_batch, 'answer': answers, 'predictions': generated_answers, 'thoughts': generated_thoughts}) + "\n")
                elif args.model == "deepseek-ai/DeepSeek-R1":
                    f.write(json.dumps({'input_batch': input_batch, 'answer': answers, 'predictions': generated_answers, 'thoughts': thoughts}) + "\n")
                else:
                    f.write(json.dumps({'input_batch': input_batch, 'answer': answers, 'predictions': generated_answers}) + "\n")

    print()
    print(args)
    print('Model: ', args.model)

    # SAVE EVAL RESULTS
    SCORES_DIR = os.path.join(PROJECT_BASE, "scores", args.tomi_set, args.run_id)
    os.makedirs(SCORES_DIR, exist_ok=True)
    ratio_per_question_type = {k: correct_per_question_type[k] / total_per_question_type[k] for k in total_per_question_type.keys()}
    score_df = pd.DataFrame([total_per_question_type, correct_per_question_type, ratio_per_question_type])
    score_df = score_df[['reality', 'memory', 'first_order_0_tom', 'first_order_1_tom', 'first_order_0_no_tom', 'first_order_1_no_tom', 'second_order_0_tom', 'second_order_1_tom', 'second_order_0_no_tom', 'second_order_1_no_tom']].T
    score_df.columns = ['Total', 'Correct', 'Accuracy']
    score_df['Accuracy'] = score_df['Accuracy'].apply(lambda x: round(x, 4) * 100)
    score_file_name = os.path.join(SCORES_DIR, f"{file_id}.json")
    qtype_counts_file_name = os.path.join(SCORES_DIR, f"qtype-freqs_{file_id}.json")
    accuracy = score_df['Accuracy'].copy()
    accuracy['first_order_false_belief'] = (accuracy['first_order_0_tom'] + accuracy['first_order_1_tom']) / 2
    accuracy['first_order_true_belief'] = (accuracy['first_order_0_no_tom'] + accuracy['first_order_1_no_tom']) / 2
    accuracy['second_order_false_belief'] = (accuracy['second_order_0_tom'] + accuracy['second_order_1_tom']) / 2
    accuracy['second_order_true_belief'] = (accuracy['second_order_0_no_tom'] + accuracy['second_order_1_no_tom']) / 2
    accuracy['false_belief'] = (accuracy['first_order_false_belief'] + accuracy['second_order_false_belief']) / 2
    accuracy['true_belief'] = (accuracy['first_order_true_belief'] + accuracy['second_order_true_belief']) / 2
    accuracy['belief'] = (accuracy['false_belief'] + accuracy['true_belief']) / 2
    accuracy['answer_prompt'] = args.answer_prompt
    accuracy['answer_prompt_with_therefore'] = args.answer_prompt_with_therefore
    print(accuracy)
    accuracy.to_json(score_file_name, indent=4)
    score_df.to_json(qtype_counts_file_name, indent=4)
    accuracy_for_wandb = accuracy.to_dict()
    del accuracy_for_wandb['answer_prompt']
    del accuracy_for_wandb['answer_prompt_with_therefore']

    wandb.log(accuracy_for_wandb)
    wandb.finish()


if __name__ == "__main__":
    parser = get_tracer_parser()
    # parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--max_questions_per_type', type=int, default=50)
    parser.add_argument('--tomi-set', type=str, default="paraphrased_tomi", help='ToMi subset to test')
    parser.add_argument('--input_file', type=str, help='Input file to test', default='test')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--model', type=str,
                        help='Model to use to answer final question.')
    parser.add_argument('--use-cot',
                        type=bool,
                        default=False,
                        help='whether to run the model with zero-shot cot',
    )
    parser.add_argument('--run-id', type=str, required=True, help='Run ID')
    parser.add_argument('--print', action='store_true', help='whether to print the outputs')
    parser.add_argument('--existing-savepoint', default=None, help='path to existing savepoint')
    parser.add_argument('--reasoning-effort', type=str, help='Reasoning effort')
    args = parser.parse_args()
    main(args)
