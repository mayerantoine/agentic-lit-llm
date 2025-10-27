# Agentic Literature Review Generation

Automated tool for generating "Related Work" sections for scientific papers using AI agents and retrieval-augmented generation (RAG). Given a research idea and a collection of paper abstracts, this tool retrieves relevant papers, scores their relevance using debate-style reasoning, and generates a cohesive literature review with proper citations.

**Based on:** "LitLLMs, LLMs for Literature Review: Are we there yet?" (Agarwal et al., 2024)

## How It Works

1. **Load** abstracts from CSV → **Index** into vector database (ChromaDB + hybrid search)
2. **Retrieve** relevant papers using semantic + keyword search (BM25)
3. **Score** papers with AI agent using structured debate-style reasoning
4. **Generate** cohesive "Related Work" section with citations

## Prerequisites

- **Python 3.13+**
- **OpenAI API key** (for GPT-4o-mini)
- **[uv](https://github.com/astral-sh/uv)** (recommended) or pip

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/mayerantoine/agentic-lit-llm.git
cd agentic-lit-llm
uv sync  # or: pip install -e .

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run the complete pipeline
python main.py run
# Follow the prompts:
#   - Enter path to CSV: rag.csv
#   - Recreate index: No (or Yes if first time)
#   - Enter your research query
```

## Usage

The CLI has three commands for different workflows:

### Command 1: `run` - Complete Pipeline

Run the entire pipeline from start to finish (indexing + generation).

```bash
python main.py run
```

**Interactive prompts:**
- CSV file path (e.g., `rag.csv`)
- Whether to recreate index
- Your research query/idea

**Options:**
```bash
python main.py run \
  --output my_review.txt \
  --hybrid-k 100 \
  --top-k 5 \
  --relevance-model gpt-4o-mini \
  --generation-model gpt-4o-mini
```

**When to use:** First time or when you want to do everything in one step.

---

### Command 2: `index` - Build Index Only

Create or update the vector database without generating a review.

```bash
python main.py index
```

**Interactive prompts:**
- CSV file path
- Whether to recreate index

**Options:**
```bash
python main.py index \
  --persist-dir ./my-index \
  --random-seed 42
```

**When to use:**
- First-time setup to build the index
- Updating index when you add new papers
- Separating indexing from generation (useful for large corpora)

---

### Command 3: `generate` - Generate from Existing Index

Generate a literature review using a pre-built index.

```bash
python main.py generate
```

**Interactive prompts:**
- CSV file path (needed for metadata)
- Your research query/idea

**Options:**
```bash
python main.py generate \
  --csv-path rag.csv \
  --output review.txt \
  --hybrid-k 50 \
  --num-score 20 \
  --top-k 3
```

**When to use:**
- Index already exists (from `index` or `run` command)
- Generate multiple reviews from same corpus
- Experiment with different queries or parameters

---

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o, --output` | Output file path | `generated_related_work.txt` |
| `-p, --persist-dir` | ChromaDB directory | `./corpus-data/chroma_db` |
| `--hybrid-k` | Papers to retrieve | `50` |
| `--num-score` | Papers to score | All retrieved |
| `--top-k` | Papers in final review | `3` |
| `--relevance-model` | Model for scoring | `gpt-4o-mini` |
| `--generation-model` | Model for generation | `gpt-4o-mini` |

## Notebook Usage

For interactive exploration and experimentation:

```bash
jupyter lab generate-related-work.ipynb
```

**The notebook provides:**
- Step-by-step execution with intermediate results
- Configurable parameters in dedicated cells
- Visualization of retrieval and scoring
- Same pipeline as CLI but with granular control

**When to use:**
- Learning how the pipeline works
- Debugging or analyzing results
- Experimenting with parameters
- Custom modifications

## Data Format

Your CSV file must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique paper identifier |
| `title` | str | Paper title |
| `abstract` | str | Paper abstract text |

**Example** (`rag.csv`):
```csv
id,title,abstract
1,"PaperQA: Retrieval-Augmented Generative Agent","Large Language Models (LLMs) generalize..."
2,"Improving Retrieval for RAG based QA","The effectiveness of Large Language Models..."
```

**Included:** `rag.csv` contains 78 biomedical RAG papers for testing.

## Output

The tool generates a text file containing:
- Your research query
- Generated "Related Work" section with citations (e.g., [1], [23], [45])
- Full references for cited papers

**Example output:**
```
RELATED WORK:
Recent advances in retrieval-augmented generation have shown promise for biomedical
literature [1]. Several approaches combine semantic and keyword search [23], while
others focus on domain-specific fine-tuning [45]...

REFERENCES:
[1] PaperQA: Retrieval-Augmented Generative Agent for Scientific Research
[23] Improving accuracy of gpt-3/4 results on biomedical data...
```

## Troubleshooting

**"No index found" error:**
```bash
# Run index command first
python main.py index
```

**MallocStackLogging warning (macOS):**
- Harmless warning, automatically suppressed in code
- If you see it, update to latest version

**Empty or poor results:**
- Increase `--hybrid-k` (retrieve more papers)
- Increase `--num-score` (score more candidates)
- Check your research query is specific enough
- Ensure CSV has relevant papers

## Dependencies

Key libraries (see `pyproject.toml` for full list):
- **typer, rich**: CLI interface
- **langchain, langchain-chroma**: RAG framework
- **chromadb**: Vector database
- **openai-agents**: Agentic workflows
- **sentence-transformers**: Embeddings (all-MiniLM-L6-v2)
- **pandas**: Data manipulation
- **pydantic**: Data validation

## Citation

```bibtex
@software{agentic_lit_llm,
  author = {Antoine Mayer},
  title = {Agentic Literature Review Generation},
  year = {2025},
  url = {https://github.com/mayerantoine/agentic-lit-llm}
}
```

## Acknowledgments

- Agarwal, Shubham et al. "LitLLMs, LLMs for Literature Review: Are we there yet?" Trans. Mach. Learn. Res. 2025 (2024)
- Built with [OpenAI Agents](https://github.com/openai/swarm) framework
- Powered by [LangChain](https://www.langchain.com/) and [ChromaDB](https://www.trychroma.com/)

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests if applicable
4. Submit a pull request

## License

See LICENSE file for details.
