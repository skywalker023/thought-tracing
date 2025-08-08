# 🚲 Thought Tracing
This is the official repository of our 2025 COLM paper:<br>
<a href="https://arxiv.org/abs/2502.11881"><b>"Thought Tracing: Hypothesis-Driven Theory-of-Mind Reasoning for Large Language Models"</b></a>

Please cite our work if you found the resources in this repository useful:

```bib
@inproceedings{kim2025tracing,
    title={Hypothesis-Driven Theory-of-Mind Reasoning for Large Language Models},
    author={Hyunwoo Kim and Melanie Sclar and Tan Zhi-Xuan and Lance Ying and Sydney Levine and Yang Liu and Joshua B. Tenenbaum and Yejin Choi},
    booktitle={COLM},
    year=2025
}
```

## Environment setup

```bash
conda env create -f environment.yml; conda activate thought-tracing
pip install flash-attn==2.5.1.post1
python -m ipykernel install --user --name thought-tracing --display-name "thought-tracing"
huggingface-cli login
```

### Adding your own agent

All you need to do is create an agent class with the method `interact()` or `batch_interact()`.

## Running Evaluation

### ToMi
```
cd revised_tomi
python test_tomi.py --model gpt-4o-2024-11-20 --use-tracing --tracing-model gpt-4o-2024-11-20 --print --run-id tracer-first-run --dataset tomi --tracer-type tracer
```

### FANToM
```
cd fantom
python test_fantom.py --model gpt-4o-2024-08-06 --use-tracing --tracing-model gpt-4o-2024-08-06 --print --dataset fantom --tracer-type multi-tracer --input-is-chat --run-id tracer-first-run
```

### BigToM
```
cd bigtom
python test_bigtom.py --model Qwen/Qwen2.5-72B-Instruct-Turbo --use-tracing --tracing-model Qwen/Qwen2.5-72B-Instruct-Turbo --dataset bigtom --print True --tracer-type tracer --target-subset agree90 --use-helper-llm --run-id tracer-first-run
```

### MMToM-QA
```
cd mmtom_qa
python test_mmtom.py --model gpt-4o --use-tracing --tracing-model gpt-4o --dataset mmtom --print --likelihood-estimate prompting --tracer-type tracer --run-id debugging_gpt-4o
```
