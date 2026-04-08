# ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents

This repository accompanies the paper: *"ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents"*. It contains the implementation of **ReviewGrounder**, a rubric-guided, tool-integrated multi-agent framework for generating substantive, evidence-grounded academic paper reviews. 

ReviewGrounder addresses the key limitation of existing LLM-based reviewers—their tendency to produce superficial, formulaic comments lacking substantive feedback—by explicitly leveraging reviewer rubrics and contextual grounding in existing work.

## System Architecture

ReviewGrounder implements a multi-agent framework with clear role separation:

### Drafting Agent (`paper_reviewer.py`)
The **drafter** generates an initial review draft based solely on the paper content. This stage produces a structured review with strengths, weaknesses, suggestions, and questions, but may lack deep contextual grounding.

### Grounding Agents

1. **Related Work Searcher** (`related_work_searcher.py`): 
   - Generates search keywords from paper content
   - Retrieves relevant papers via academic APIs
   - Summarizes and analyzes related work
   - Provides context for novelty assessment

2. **Paper Results Analyzer** (`paper_results_analyzer.py`):
   - Extracts and analyzes experimental sections
   - Summarizes experimental setup, results, and findings
   - Identifies limitations and gaps

3. **Paper Insight Miner** (`paper_insight_miner.py`):
   - Extracts key insights and contributions
   - Identifies technical strengths and weaknesses

4. **Review Refiner** (`review_refiner.py`):
   - Synthesizes information from all grounding agents
   - Refines the initial draft with evidence-based critiques
   - Ensures suggestions are actionable and well-justified
   - Maintains consistency across review sections

### Evaluation System (`src/evaluator/`)
The **ReviewBench** evaluation framework:
- **Rubric Generation**: Creates paper-specific rubrics from venue guidelines, paper content, and human reviews
- **LLM-based Evaluation**: Deep qualitative assessment aligned with rubrics
- **Rule-based Metrics**: Quantitative metrics (MSE, MAE, Spearman correlation)

## Installation

### Prerequisites

- Python >= 3.8
- CUDA-capable GPU (for local vLLM deployment, optional if using OpenAI API)
- Sufficient GPU memory for your chosen model (if using vLLM)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd ReviewGrounder
```

2. Install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

3. Configure your API keys and settings:
   - Copy `shared/configs/config.yaml` and customize as needed
   - Set environment variables:
     - `ASTA_API_KEY`: For paper search via Asta API (recommended)
     - `OPENAI_API_KEY`: If using OpenAI API instead of vLLM
     - `S2_API_KEY`: Alternative paper search API (optional)

4. (Optional) If using local vLLM, start your vLLM service:
```bash
# Start vLLM service on a single port
bash scripts/gpt_oss_start_vllm_service.sh

# Or start multiple services with load balancing
bash scripts/start_vllm_with_balancer.sh
```

## Quick Start

### Basic Usage

Generate a review using the command-line interface:

```bash
python -m src.reviewer_agent.cli --paper paper.json --output review.json
```

Where `paper.json` contains your paper data in JSON format with fields like `title`, `abstract`, `text`, etc.

### Using the Python API

For programmatic access:

```python
from src.reviewer_agent import review_paper_with_refiner

# Load your paper data
paper_data = {
    "title": "Your Paper Title",
    "abstract": "Paper abstract...",
    "text": "Full paper text...",
    # ... other fields
}

# Generate review (drafting + grounding stages)
review = review_paper_with_refiner(paper_data=paper_data)
print(review)
```

The `review_paper_with_refiner` function implements the full ReviewGrounder pipeline:
1. **Drafting**: Generates initial review draft
2. **Grounding**: Retrieves related work, analyzes results, extracts insights
3. **Refinement**: Synthesizes all information into a refined, evidence-grounded review

## Usage Examples

### Generate a Review with Related Work Context

```bash
python -m src.reviewer_agent.cli \
    --paper paper.json \
    --max-related-papers 15 \
    --review-format detailed \
    --output review.json
```

### Filter Related Work by Date and Venue

```bash
python -m src.reviewer_agent.cli \
    --paper paper.json \
    --publication-date-range "2020:" \
    --venues "ICLR,NeurIPS,ICML" \
    --output review.json
```

### Use Custom vLLM Endpoint

```bash
python -m src.reviewer_agent.cli \
    --paper paper.json \
    --vllm-url "http://your-server:8000/v1" \
    --output review.json
```

### Evaluate Reviews on ReviewBench

```python
# 1. Generate reviews
from src.reviewer_agent import review_paper_with_refiner
review = review_paper_with_refiner(paper_data={...})

# 2. Evaluate reviews using ReviewBench
from src.evaluator import evaluate_reviews
results = evaluate_reviews(parquet_path="reviews.parquet")
```

## Directory Structure

```
anonymize_codebase/
├── src/
│   ├── reviewer_agent/          # ReviewGrounder implementation
│   │   ├── __init__.py
│   │   ├── paper_reviewer.py    # Drafting agent
│   │   ├── review_refiner.py    # Grounding agent: review refinement
│   │   ├── related_work_searcher.py  # Grounding agent: literature search
│   │   ├── paper_results_summarizer.py  # Grounding agent: results analysis
│   │   ├── paper_insight_miner.py  # Grounding agent: insight extraction
│   │   ├── main_pipeline.py     # Full pipeline orchestration
│   │   ├── cli.py               # Command-line interface
│   │   └── paper_search/        # Paper search APIs
│   │       ├── asta_api.py
│   │       ├── semantic_scholar_api.py
│   │       └── paper_retriever.py
│   │
│   └── evaluator/               # ReviewBench evaluation framework
│       ├── 1_get_rubrics.py     # Rubric generation
│       ├── 2_evaluate.py        # Review evaluation
│       └── ...
│
├── shared/
│   ├── utils/                   # Shared utilities
│   │   ├── llm_service.py       # LLM service abstraction
│   │   ├── load_balancer.py     # Load balancing for vLLM
│   │   ├── reranker.py          # Paper reranking
│   │   └── ...
│   │
│   └── configs/                 # Configuration files
│       ├── config.yaml          # Main config
│       ├── llm_service_config.yaml  # LLM service settings
│       └── prompts.yaml         # Review generation prompts
│
├── scripts/                      # Utility scripts
│   ├── start_vllm_with_balancer.sh
│   ├── start_load_balancer.sh
│   └── ...
│
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Configuration Guide

### LLM Service Configuration

ReviewGrounder supports two LLM backends:

1. **vLLM** (recommended for local deployment): Fast inference with local GPU
   - Default: GPT-OSS-120B for grounding stage
   - Can use smaller models (e.g., Phi-4-14B) for drafting stage

2. **OpenAI API**: Cloud-based, no local GPU required

Configure in `shared/configs/llm_service_config.yaml`:

```yaml
vllm:
  base_url: "http://localhost:8000/"
  model_name: "openai/gpt-oss-120b"
  max_tokens: 16384

gpt:
  enabled: false
  api_key: "your-api-key-here"
  model_name: "gpt-4o"
```

We offer the option of assigning different backends for each agent.
```yaml
llm_assignments:
  keyword_generator: "vllm"  # For related work search
  paper_summarizer: "vllm"   # For results summarization
  reviewer: "vllm"           # For drafting stage
  refiner: "vllm"            # For grounding/refinement stage
```

### Paper Search Configuration

Configure paper search APIs in `shared/configs/config.yaml`:

```yaml
paper_search:
  asta:
    api_key: null  # Set via ASTA_API_KEY env var
    endpoint: "https://asta-tools.allen.ai/mcp/v1"
  
  semantic_scholar:
    api_key: null  # Set via S2_API_KEY env var
```

### Review Format Options

Choose from different review formats:
- `detailed`: Comprehensive review with all sections (default)
- `summary`: Concise review summary
- `structured`: Structured format with specific sections
- `strict_detailed`: Strict adherence to detailed format requirements

## Load Balancing for vLLM

For production use with multiple GPUs, you can set up load balancing:

```bash
# Start 4 vLLM services on ports 8000-8003
bash scripts/gpt_oss_start_vllm_service.sh

# Start load balancer on port 8004
python -m shared.utils.load_balancer \
    --backends http://localhost:8000/v1 http://localhost:8001/v1 http://localhost:8002/v1 http://localhost:8003/v1 \
    --port 8004 \
    --strategy round_robin
```

Then point your config to `http://localhost:8004/v1`.

## Evaluation: ReviewBench

ReviewGrounder is evaluated on **ReviewBench**, a benchmark that:

- Leverages paper-specific rubrics derived from:
  - Official venue guidelines (e.g., ACL, ICML, NeurIPS, ICLR)
  - Paper content
  - Human-written reviews

- Evaluates reviews across diverse dimensions:
  - Evidence-based critique
  - Constructive tone
  - Technical depth
  - And more...

- Measures both:
  - Alignment with human judgments (scores, decisions)
  - Rubric-based quality (beyond just outcome prediction)

See `src/evaluator/` for the evaluation framework implementation.

## Citation

If you use ReviewGrounder in your research, please cite:

```bibtex
@inproceedings{reviewgrounder2026,
  title={ReviewGrounder: Improving Review Substantiveness with Rubric-Guided, Tool-Integrated Agents},
  author={Anonymous},
  booktitle={Proceedings of ACL 2026},
  year={2026}
}
```
