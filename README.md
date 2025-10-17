# Agentic Literature Review Generation

Implementing Lit-LLM: An Agentic Workflow for Synthesizing Research Paper Literature Reviews using OpenAI Agents.
We reproduced and adapted the multi-document summarization approach from the paper "LitLLMs, LLMs for Literature Review: Are we there yet?" (2024) for our own use and testing.

The general workflow starts with a paper idea and research questions. You then use keywords to search several databases and download abstracts related to background work, previous research, and gaps connected to your research question.

Given your research idea and list of abstracts, this tool quickly generates comprehensive "Related Work" sections by intelligently retrieving, scoring, and synthesizing insights from your abstracts.


### Pipeline Flow

1. **Data Loading**: Load abstracts from CSV (requires: `id`, `title`, `abstract`)
2. **Vector Store Initialization**: Create/load ChromaDB index with embeddings
3. **Hybrid Retrieval**: Retrieve top-k relevant papers using semantic + keyword search
4. **Relevance Scoring**: AI agent scores papers using **debate-style reasoning**
5. **Top-K Selection**: Select most relevant papers by probability score
6. **Related Work Generation**: AI agent generates cohesive literature review text

## Installation

### Prerequisites

- Python 3.13+
- OpenAI API key (for GPT-4o-mini)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mayerantoine/agentic-lit-llm.git
   cd agentic-lit-llm
   ```

2. **Install dependencies**:
   ```bash
   # Using uv (recommended)
   uv sync

   # Or using pip
   pip install -e .
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

4. **Prepare your data**:
   - CSV file with columns: `id`, `title`, `abstract`
   - Place in accessible location (e.g., `abstracts_rag.csv`)

## Usage

### CLI Application

The CLI provides an interactive interface for generating literature reviews:

```bash
python main.py run
```

**Interactive Prompts**:
- Enter path to your CSV file containing abstracts
- Enter your research query/idea (multi-line supported)

**CLI Options**:
```bash
python main.py run --help

Options:
  -o, --output TEXT              Output file path [default: generated_related_work.txt]
  -p, --persist-dir TEXT         ChromaDB persist directory [default: ./corpus-data/chroma_db]
  --recreate-index              Recreate index from scratch [default: False]
  --hybrid-k INTEGER            Number of papers to retrieve [default: 50]
  --num-score INTEGER           Number of papers to score (None = all)
  --top-k INTEGER               Number of top papers for related work [default: 3]
  --relevance-model TEXT        Model for relevance scoring [default: gpt-4o-mini]
  --generation-model TEXT       Model for text generation [default: gpt-4o-mini]
```

**Example**:
```bash
python main.py run \
  --output my_literature_review.txt \
  --hybrid-k 100 \
  --num-score 20 \
  --top-k 5 \
  --relevance-model gpt-4o-mini \
  --generation-model gpt-4o-mini
```

### Jupyter Notebook

For interactive exploration and experimentation:

```bash
jupyter lab generate-related-work.ipynb
```

The notebook provides:
- Step-by-step pipeline execution
- Configurable parameters in dedicated cells
- Visualization of retrieval and scoring results
- Easy experimentation with different configurations


## Data Format

Your CSV file must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique identifier for each paper |
| `title` | str | Paper title |
| `abstract` | str | Paper abstract text |

**Example CSV**:
```csv
id,title,abstract
1,"Deep Learning for NLP","This paper presents..."
2,"Transformers in Action","We introduce a novel approach..."
```


## Dependencies

Key libraries:
- **typer**: CLI framework
- **rich**: Terminal UI
- **langchain**: RAG framework
- **chromadb**: Vector database
- **openai-agents**: Agentic workflows
- **sentence-transformers**: Embeddings
- **pandas**: Data manipulation
- **pydantic**: Data validation

See `pyproject.toml` for complete list.


## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Citation

If you use this tool in your research, please cite:

```bibtex
@software{agentic_lit_llm,
  author = {Antoine Mayer},
  title = {Agentic Literature Review Generation},
  year = {2025},
  url = {https://github.com/mayerantoine/agentic-lit-llm}
}
```

## Acknowledgments

- Agarwal, Shubham et al. “LitLLMs, LLMs for Literature Review: Are we there yet?” Trans. Mach. Learn. Res. 2025 (2024): n. pag.
- Built with [OpenAI Agents](https://github.com/openai/swarm) framework
- Powered by [LangChain](https://www.langchain.com/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- UI powered by [Rich](https://rich.readthedocs.io/)

## Contact

For questions or issues, please open an issue on GitHub.
