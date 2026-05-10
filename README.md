<a name="readme-top"></a>

<h1 align="center">ReviewGrounder</h1>

<h2 align="center">
  Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents
</h2>

<div align="center">
  <a href="https://arxiv.org/abs/2604.14261v1"><img src="https://img.shields.io/badge/arXiv-B31B1B?style=for-the-badge&logo=arXiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/spaces/ReviewGrounder/GradioDemo"><img src="https://img.shields.io/badge/Demo-F97316.svg?style=for-the-badge&logo=gradio&logoColor=white" alt="Demo"></a>
  <a href="https://huggingface.co/ReviewGrounder"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face"></a>
  <a href="https://x.com/yuz9yuz/status/2048823620193902704"><img src="https://img.shields.io/badge/Twitter-000000?style=for-the-badge&logo=X&logoColor=white" alt="Twitter"></a>
</div>

---

## Introduction

**ReviewGrounder** is a **rubric-guided, tool-integrated agent framework** for generating substantive academic paper reviews. It targets a common failure mode of LLM-based reviewing: reviews that are fluent but generic, shallow, or weakly grounded in the paper and surrounding literature.

Instead of asking a single model to produce a final review in one pass, ReviewGrounder decomposes reviewing into a drafting stage and multiple grounding stages. The system first writes an initial review, then gathers additional evidence from paper results, paper insights, and related work before producing a refined, evidence-grounded review.

This repository contains the ReviewGrounder implementation, the Gradio demo entry point, paper-search utilities, configurable LLM backends, and the ReviewBench evaluation code used to assess review quality with paper-specific rubrics.

## Main Results

ReviewGrounder is evaluated with **ReviewBench**, a rubric-based evaluation framework designed to measure whether reviews are substantive, evidence-backed, and useful for authors. ReviewBench combines venue guidelines, paper content, and human reviews to build paper-specific rubrics, then evaluates generated reviews against those rubrics.

With a Phi-4-14B-based drafter and a GPT-OSS-120B-based grounding stage, ReviewGrounder outperforms strong foundation-model, agentic-reviewing, and fine-tuned-reviewer baselines on ReviewBench, including larger backbones such as GPT-4.1 and DeepSeek-R1-670B, across both human-judgment alignment and rubric-based review quality.

The full experimental setup and results are available in the paper:

- **Paper:** [ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents](https://arxiv.org/abs/2604.14261v1)
- **Demo:** [ReviewGrounder Gradio Demo](https://huggingface.co/spaces/ReviewGrounder/GradioDemo)
- **Models and artifacts:** [ReviewGrounder on Hugging Face](https://huggingface.co/ReviewGrounder)

## Key Features

- **Rubric-guided review refinement:** ReviewGrounder uses explicit reviewing criteria to move beyond generic praise and criticism toward targeted, actionable feedback.
- **Tool-integrated grounding:** The pipeline augments the initial review with related-work retrieval, experimental-results analysis, and paper-insight mining.
- **Multi-agent role separation:** Dedicated agents handle drafting, literature grounding, result analysis, insight extraction, and final refinement.
- **Flexible LLM backends:** Components can be assigned to different OpenAI-compatible backends, including local vLLM services and API-hosted models.
- **Interactive demo support:** The repository includes a Gradio application for PDF upload, stepwise review generation, and raw JSON inspection.
- **ReviewBench evaluation:** The evaluator supports rubric generation, LLM-based review assessment, and quantitative agreement metrics.

---

## Table of Contents

- [Setup](#setup)
- [Quick Start](#quick-start)
- [ReviewGrounder Pipeline](#reviewgrounder-pipeline)
- [Configuration](#configuration)
- [ReviewBench Evaluation](#reviewbench-evaluation)
- [Repository Layout](#repository-layout)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

<a name="setup"></a>
## Setup

### Prerequisites

- Python 3.8+
- `uv` or `pip`
- `ASTA_API_KEY` for related-paper search through Asta
- One OpenAI-compatible LLM endpoint, such as a local vLLM server or hosted API
- CUDA-capable GPUs if you run local vLLM models or local rerankers

### Installation

```bash
git clone <repository-url>
cd ReviewGrounder

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

If you prefer `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment Variables

Set the API keys required by your configuration:

```bash
export ASTA_API_KEY="your-asta-api-key"
export OPENAI_API_KEY="your-openai-api-key"  # optional, if using an OpenAI-compatible hosted backend
export S2_API_KEY="your-semantic-scholar-api-key"  # optional fallback paper-search backend
```

ReviewGrounder reads model and retrieval settings from:

- [`shared/configs/config.yaml`](shared/configs/config.yaml)
- [`shared/configs/llm_service_config.yaml`](shared/configs/llm_service_config.yaml)
- [`shared/configs/prompts.yaml`](shared/configs/prompts.yaml)

<a name="quick-start"></a>
## Quick Start

### Try the Hosted Demo

The fastest way to try ReviewGrounder is the Hugging Face Space:

[https://huggingface.co/spaces/ReviewGrounder/GradioDemo](https://huggingface.co/spaces/ReviewGrounder/GradioDemo)

Upload a paper PDF, provide an OpenAI-compatible API endpoint if needed, and inspect the stepwise outputs for the initial review, related work, result analysis, insight mining, and final refined review.

### Run the Local Gradio App

```bash
export ASTA_API_KEY="your-asta-api-key"
python app.py
```

By default, Gradio will print a local URL such as `http://127.0.0.1:7860`.

### Review a Paper from the CLI

Prepare a JSON file with fields such as `title`, `abstract`, `content` or `text`, and optional `keywords`:

```json
{
  "title": "Your Paper Title",
  "abstract": "Paper abstract...",
  "content": "Full paper text...",
  "keywords": ["review generation", "scientific evaluation"]
}
```

Then run:

```bash
python -m src.reviewer_agent.cli \
  --paper paper.json \
  --output review.json \
  --verbose
```

Useful options:

```bash
python -m src.reviewer_agent.cli \
  --paper paper.json \
  --max-related-papers 15 \
  --publication-date-range "2020:" \
  --venues "ICLR,NeurIPS,ICML" \
  --review-format detailed \
  --output review.json
```

### Use the Python API

```python
from src.reviewer_agent import review_paper_with_refiner

paper_data = {
    "title": "Your Paper Title",
    "abstract": "Paper abstract...",
    "content": "Full paper text...",
    "keywords": ["paper review", "LLM agents"],
}

review = review_paper_with_refiner(paper_data=paper_data)
print(review)
```

<a name="reviewgrounder-pipeline"></a>
## ReviewGrounder Pipeline

ReviewGrounder uses a drafting-and-grounding workflow:

| Stage | Component | File | Role |
|-------|-----------|------|------|
| Drafting | Paper Reviewer | `src/reviewer_agent/paper_reviewer.py` | Generates the initial review from the target paper. |
| Related-work grounding | Related Work Searcher | `src/reviewer_agent/related_work_searcher.py` | Searches, reranks, and summarizes relevant prior work. |
| Results grounding | Paper Results Analyzer | `src/reviewer_agent/paper_results_analyzer.py` | Extracts experimental evidence, claims, and limitations. |
| Insight grounding | Paper Insight Miner | `src/reviewer_agent/paper_insight_miner.py` | Identifies core contributions, technical insights, and weaknesses. |
| Refinement | Review Refiner | `src/reviewer_agent/review_refiner.py` | Synthesizes all evidence into the final grounded review. |

The high-level orchestration lives in:

- `src/reviewer_agent/main_pipeline.py`
- `src/reviewer_agent/main_pipeline_concurrent.py`
- `src/reviewer_agent/single_paper_inference.py`

## Configuration

### LLM Backends

ReviewGrounder supports OpenAI-compatible model services. You can assign different model backends to different pipeline components in [`shared/configs/llm_service_config.yaml`](shared/configs/llm_service_config.yaml):

```yaml
llm_assignments:
  insight_miner: "vllm_oss"
  results_analyzer: "vllm_oss"
  reviewer: "vllm_deepreviewer"
  keyword_generator: "vllm_oss"
  paper_summarizer: "vllm_oss"
  refiner: "vllm_oss"
```

Each backend defines its own `base_url`, `model_name`, sampling parameters, timeout, and retry behavior.

### Local vLLM Services

Start a single local vLLM service:

```bash
bash scripts/gpt_oss_start_vllm_service.sh
```

For multi-GPU serving with load balancing:

```bash
bash scripts/start_vllm_with_balancer.sh
```

Then set the corresponding backend `base_url` in `shared/configs/llm_service_config.yaml`.

### Paper Search and Reranking

Paper search is configured in [`shared/configs/config.yaml`](shared/configs/config.yaml):

```yaml
paper_search:
  asta:
    api_key: null
    api_key_pool_path: "asta_api_pool.txt"
    endpoint: "https://asta-tools.allen.ai/mcp/v1"
  semantic_scholar:
    api_key: null
  reranker:
    model: "OpenScholar/OpenScholar_Reranker"
```

At runtime, the Asta key can be supplied through `ASTA_API_KEY` or `--asta-api-key`.

<a name="reviewbench-evaluation"></a>
## ReviewBench Evaluation

ReviewBench evaluates generated reviews with paper-specific rubrics. The evaluation framework includes:

- **Rubric generation:** builds rubrics from venue guidelines, paper content, and human reviews.
- **LLM-based review assessment:** scores generated reviews against substantive, rubric-aligned criteria.
- **Rule-based metrics:** computes agreement and error metrics such as MSE, MAE, and Spearman correlation.

The evaluator code is under [`src/evaluator/`](src/evaluator/). A typical two-step workflow is:

```bash
python src/evaluator/1_get_rubrics.py \
  --json_path input_reviews.json \
  --output_path eval_rubrics.json \
  --yaml_path src/evaluator/prompts.yaml \
  --config_path src/evaluator/configs.yaml \
  --max_workers 5

python src/evaluator/2_evaluate.py \
  --rubrics_path eval_rubrics.json \
  --reviews_path model_reviews.json \
  --mode both \
  --yaml_path src/evaluator/prompts.yaml \
  --config_path src/evaluator/configs.yaml \
  --semantic_output semantic_results.json \
  --auto_metric_output auto_metric_results.json \
  --max_workers 32
```

The repository also includes evaluator scripts for multiple review-generation baselines:

```text
src/evaluator/2_evaluate.py
src/evaluator/2_evaluate_agenticreview.py
src/evaluator/2_evaluate_aiscientist.py
src/evaluator/2_evaluate_cyclereviewer.py
```

<a name="repository-layout"></a>
## Repository Layout

```text
ReviewGrounder/
├── app.py                         # Gradio app entry point
├── gradio_app/                    # Gradio UI components and PDF inference helpers
├── src/
│   ├── reviewer_agent/            # ReviewGrounder pipeline
│   │   ├── cli.py                 # Command-line interface
│   │   ├── main_pipeline.py       # Main pipeline orchestration
│   │   ├── paper_reviewer.py      # Initial review drafter
│   │   ├── related_work_searcher.py
│   │   ├── paper_results_analyzer.py
│   │   ├── paper_insight_miner.py
│   │   ├── review_refiner.py
│   │   └── paper_search/          # Asta and Semantic Scholar integrations
│   └── evaluator/                 # ReviewBench evaluation
├── shared/
│   ├── configs/                   # Model, retrieval, and prompt configs
│   └── utils/                     # LLM service, reranking, logging, and parsing utilities
├── scripts/                       # vLLM, reranker, and load-balancer helpers
├── requirements.txt
└── README.md
```

<a name="acknowledgements"></a>
## Acknowledgements

ReviewGrounder builds on open-source tools and services for LLM inference, academic paper retrieval, reranking, and interactive demos, including vLLM, Gradio, Asta, Semantic Scholar, OpenAI-compatible APIs, and Hugging Face.

<a name="citation"></a>
## Citation

If you use ReviewGrounder in your research, please cite:

```bibtex
@misc{li2026reviewgrounder,
  title={ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents},
  author={Zhuofeng Li and Yi Lu and Dongfu Jiang and Haoxiang Zhang and Yuyang Bai and Chuan Li and Yu Wang and Shuiwang Ji and Jianwen Xie and Yu Zhang},
  year={2026},
  eprint={2604.14261},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
