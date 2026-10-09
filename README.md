# System One models as rerankers

Zero-shot reranking of the BM25 top-100 of TREC DL 2019 and 2020 with System One models
(Jev, Luna, kev, decider, winnow) and the LLMs they were fine-tuned from (GPT-6 Luna,
Qwen3.5-4B-Base, Gemma 4 E4B-it), using pointwise, listwise and pairwise reranking.

```
rerank.py       the whole experiment: data, questions, methods, models, evaluation
results/runs/   the final run files: <dataset>.<model>.<method>.k100.run
```

## Setup

```bash
uv sync --extra hf          # without --extra hf for the API models only
cp .env.example .env        # add TYPESAFE_API_KEY (Jev) and OPENROUTER_API_KEY (Luna, GPT-6 Luna)
```

kev, decider and winnow run on a local Ollaya server: `ollaya pull kev:4b`,
`ollaya pull decider:4b`, `ollaya pull winnow:e4b`, then `ollaya serve` (address in
`OLLAYA_URL`, default `http://127.0.0.1:11435`). Qwen3.5-4B-Base and Gemma 4 E4B-it run with
Hugging Face transformers on a GPU.

## Run

```bash
uv run python rerank.py --models jev --methods pointwise,listwise,duo --datasets dl19,dl20
```

`--models` takes `jev`, `luna`, `gpt-6-luna`, `kev`, `decider`, `winnow`, `qwen3.5-4b-base`
and `gemma4-e4b-it`. Run files are written to `results/runs/`, and the metrics are printed at
the end. Every model answer is stored in a local SQLite cache (`results/cache/answers.sqlite`,
created on the first run), so an interrupted run resumes where it stopped.
