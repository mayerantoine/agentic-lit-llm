#!/usr/bin/env python3
"""
Automated Literature Review Generation CLI

This CLI application automates the generation of "Related Work" sections for scientific papers
using an agentic RAG (Retrieval-Augmented Generation) pipeline.
"""

import os

# Suppress macOS malloc stack logging warning (harmless but noisy on Darwin/ARM)
# This must be set before any memory-intensive operations
if os.environ.get("MallocStackLogging"):
    os.environ["MallocStackLogging"] = ""

# Set tokenizers parallelism to false to avoid fork warnings
# This must be set before importing any libraries that use HuggingFace tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
from pathlib import Path
from typing import Optional, Annotated, List

import pandas as pd
import typer
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables from .env file
# This ensures OPENAI_API_KEY and other config is available for the CLI
load_dotenv(override=True)

from pipeline import (
    PipelineConfig,
    PipelineResult,
    IndexResult,
    LiteratureReviewPipeline,
    ValidationError,
    ProcessingError,
)

# Initialize Typer app and Rich console
app = typer.Typer(help="Automated Literature Review Generation using Agentic RAG")
console = Console()


def print_status(message: str, style: str = "bold blue"):
    """Print a status message."""
    console.print(f"[{style}]{message}[/{style}]")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[bold red]ERROR: {message}[/bold red]")


def print_success(message: str):
    """Print a success message."""
    console.print(f"[bold green]{message}[/bold green]")


def print_section_header(title: str):
    """Print a section header."""
    console.print()
    console.print(Panel(title, style="bold cyan"))


def extract_citations(generated_text: str) -> List[int]:
    """
    Extract unique citation IDs from generated text.

    Args:
        generated_text: Text containing citations in [id] format

    Returns:
        Sorted list of unique citation IDs
    """
    citations = re.findall(r'\[(\d+)\]', generated_text)
    return sorted(set(int(c) for c in citations))


def get_multiline_input(prompt: str) -> str:
    """Get multi-line input from user. Ends on double newline or Ctrl+D."""
    console.print(f"[bold cyan]{prompt}[/bold cyan]")
    console.print("[dim](Press Enter twice when done, or Ctrl+D on Unix/Ctrl+Z on Windows)[/dim]")

    lines = []
    empty_line_count = 0

    try:
        while True:
            try:
                line = input("> ")
                if line.strip() == "":
                    empty_line_count += 1
                    if empty_line_count >= 2:
                        break
                    lines.append(line)
                else:
                    empty_line_count = 0
                    lines.append(line)
            except EOFError:
                break
    except KeyboardInterrupt:
        console.print("\n[bold red]Input cancelled[/bold red]")
        raise typer.Exit(1)

    # Remove trailing empty lines
    while lines and lines[-1].strip() == "":
        lines.pop()

    result = "\n".join(lines).strip()

    if result:
        console.print(f"[dim]Query received ({len(result)} characters)[/dim]")

    return result


def prompt_for_csv_path() -> str:
    """Interactively prompt for CSV file path with validation."""
    while True:
        csv_path = typer.prompt("\nEnter path to CSV file containing abstracts")

        # Check if file exists
        if not Path(csv_path).exists():
            print_error(f"File not found: {csv_path}")
            retry = typer.confirm("Would you like to try another path?", default=True)
            if not retry:
                raise typer.Exit(1)
            continue

        # Check if it's a CSV file
        if not csv_path.lower().endswith('.csv'):
            print_error("File must be a CSV file (.csv)")
            retry = typer.confirm("Would you like to try another path?", default=True)
            if not retry:
                raise typer.Exit(1)
            continue

        return csv_path


def prompt_for_recreate_index() -> bool:
    """Interactively prompt whether to recreate the index."""
    return typer.confirm(
        "\nDo you want to recreate the index from scratch?",
        default=False
    )


def prompt_for_research_query() -> str:
    """Interactively prompt for research query with validation."""
    while True:
        query = get_multiline_input("\nEnter your research query/idea:")

        if not query or len(query.strip()) == 0:
            print_error("Query cannot be empty")
            retry = typer.confirm("Would you like to try again?", default=True)
            if not retry:
                raise typer.Exit(1)
            continue

        return query.strip()


def save_output(
    query: str,
    generated_text: str,
    top_k_abstracts: pd.DataFrame,
    output_path: str
):
    """Save the generated related work to a file."""
    print_status(f"Saving output to: {output_path}")

    try:
        # Extract citations using helper function
        unique_citations = extract_citations(generated_text)

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AUTOMATED LITERATURE REVIEW GENERATION\n")
            f.write("=" * 80 + "\n\n")

            f.write("RESEARCH QUERY:\n")
            f.write("-" * 80 + "\n")
            f.write(query)
            f.write("\n\n" + "=" * 80 + "\n\n")

            f.write("RELATED WORK:\n")
            f.write("-" * 80 + "\n")
            f.write(generated_text)
            f.write("\n\n" + "=" * 80 + "\n\n")

            f.write("REFERENCES:\n")
            f.write("-" * 80 + "\n")
            for paper_id in unique_citations:
                paper = top_k_abstracts[top_k_abstracts['id'] == paper_id]
                if not paper.empty:
                    f.write(f"[{paper_id}] {paper.iloc[0]['title']}\n")
                    f.write(f"    {paper.iloc[0]['abstract'][:200]}...\n\n")

            f.write("=" * 80 + "\n")

        print_success(f"Output saved to {output_path}")

    except Exception as e:
        print_error(f"Failed to save output: {str(e)}")


def display_results(generated_text: str, top_k_abstracts: pd.DataFrame):
    """Display the final results."""
    print_section_header("GENERATED RELATED WORK SECTION")

    console.print(generated_text)
    console.print()

    # Extract and display citations using helper function
    unique_citations = extract_citations(generated_text)

    console.print(f"\nCited Papers:")
    for paper_id in unique_citations:
        paper = top_k_abstracts[top_k_abstracts['id'] == paper_id]
        if not paper.empty:
            console.print(f"  [{paper_id}] {paper.iloc[0]['title']}")


@app.command()
def index(
    persist_dir: Annotated[str, typer.Option("--persist-dir", "-p", help="ChromaDB persist directory")] = "./corpus-data/chroma_db",
    random_seed: Annotated[int, typer.Option("--random-seed", help="Random seed for data shuffling")] = 42,
):
    """
    Build or update the vector store index from CSV abstracts.

    This command performs the indexing phase only (Steps 1-3):
    1. Load abstracts from CSV
    2. Initialize vector store
    3. Chunk and index documents

    Use this to create or update your index database without generating any literature reviews.
    """

    print_section_header("INDEX MANAGEMENT")

    console.print("\n[bold]Let's build/update your vector store index.[/bold]\n")

    # Prompt for CSV path
    csv_path = prompt_for_csv_path()

    # Prompt for index recreation
    recreate_index = prompt_for_recreate_index()

    # Create pipeline configuration (minimal config for indexing only)
    config = PipelineConfig(
        persist_directory=persist_dir,
        recreate_index=recreate_index,
        random_seed=random_seed,
    )

    # Configuration summary
    console.print(f"\nConfiguration:")
    console.print(f"  CSV file: {csv_path}")
    console.print(f"  Persist directory: {config.persist_directory}")
    console.print(f"  Recreate index: {config.recreate_index}")
    console.print(f"  Random seed: {config.random_seed}")

    # Execute indexing
    try:
        with console.status("[bold cyan]Building index...[/bold cyan]", spinner="dots") as status:
            pipeline = LiteratureReviewPipeline(config)
            result = pipeline.build_index(csv_path)

        console.print()
        print_success("Index building completed!")
        console.print()

        # Display results
        print_section_header("INDEXING RESULTS")
        console.print(f"CSV Path: {result.csv_path}")
        console.print(f"Total Abstracts: {result.total_abstracts}")
        console.print(f"Chunks Created: {result.chunks_created}")
        console.print(f"Total Indexed: {result.total_indexed}")
        console.print(f"Persist Directory: {result.persist_directory}")
        console.print(f"Index Recreated: {result.recreated}")

        # Final message
        console.print()
        print_success(f"Index is ready at: {result.persist_directory}")
        console.print("\n[dim]You can now use the 'generate' command to create literature reviews from this index.[/dim]")

    except ValidationError as e:
        print_error(f"Validation error: {str(e)}")
        raise typer.Exit(1)
    except ProcessingError as e:
        print_error(f"Processing error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def generate(
    output: Annotated[str, typer.Option("--output", "-o", help="Output file path")] = "generated_related_work.txt",
    persist_dir: Annotated[str, typer.Option("--persist-dir", "-p", help="ChromaDB persist directory")] = "./corpus-data/chroma_db",
    csv_path_arg: Annotated[Optional[str], typer.Option("--csv-path", "-c", help="CSV file path (required for loading abstracts)")] = None,
    hybrid_k: Annotated[int, typer.Option("--hybrid-k", help="Number of papers to retrieve")] = 50,
    num_score: Annotated[Optional[int], typer.Option("--num-score", help="Number of papers to score (None = all)")] = None,
    top_k: Annotated[int, typer.Option("--top-k", help="Number of top papers for related work")] = 3,
    relevance_model: Annotated[str, typer.Option("--relevance-model", help="Model for relevance scoring")] = "gpt-4o-mini",
    generation_model: Annotated[str, typer.Option("--generation-model", help="Model for text generation")] = "gpt-4o-mini",
):
    """
    Generate a literature review from an existing index.

    This command performs the generation phase only (Steps 4-7):
    4. Retrieve relevant papers using hybrid search
    5. Score papers for relevance
    6. Select top-k papers
    7. Generate related work section

    Requires that an index already exists (created via 'index' or 'run' command).
    You must provide the CSV path to load abstract metadata.
    """

    print_section_header("LITERATURE REVIEW GENERATION")

    console.print("\n[bold]Let's generate your literature review from the existing index.[/bold]\n")

    # Check if index exists first (fail fast before loading CSV)
    persist_path = Path(persist_dir)
    chroma_db_exists = (persist_path / "chroma.sqlite3").exists()

    if not chroma_db_exists:
        print_error(f"No index found at {persist_dir}")
        console.print("\n[dim]You need to create an index first using one of these commands:[/dim]")
        console.print("  [cyan]python main.py index[/cyan]  - Build index only")
        console.print("  [cyan]python main.py run[/cyan]    - Build index and generate review")
        raise typer.Exit(1)

    # Prompt for CSV path if not provided as argument
    if csv_path_arg is None:
        csv_path = prompt_for_csv_path()
    else:
        csv_path = csv_path_arg

    # Prompt for research query
    query = prompt_for_research_query()

    # Display query
    console.print(f"\n[bold]Research Query:[/bold]")
    console.print(Panel(query, style="dim"))

    # Create pipeline configuration
    config = PipelineConfig(
        persist_directory=persist_dir,
        recreate_index=False,  # Never recreate when generating
        hybrid_k=hybrid_k,
        num_abstracts_to_score=num_score,
        top_k=top_k,
        relevance_model=relevance_model,
        generation_model=generation_model,
    )

    # Configuration summary
    console.print(f"\nConfiguration:")
    console.print(f"  CSV file: {csv_path}")
    console.print(f"  Output file: {output}")
    console.print(f"  Persist directory: {config.persist_directory}")
    console.print(f"  Hybrid retrieval k: {config.hybrid_k}")
    console.print(f"  Papers to score: {'All' if config.num_abstracts_to_score is None else config.num_abstracts_to_score}")
    console.print(f"  Top-k papers: {config.top_k}")
    console.print(f"  Relevance model: {config.relevance_model}")
    console.print(f"  Generation model: {config.generation_model}")

    # Execute pipeline
    try:
        with console.status("[bold cyan]Running generation pipeline...[/bold cyan]", spinner="dots") as status:
            pipeline = LiteratureReviewPipeline(config)

            # Load abstracts (without re-indexing)
            pipeline.load_abstracts_only(csv_path)

            # Generate the review from existing index
            result = pipeline.generate_review(query)

        console.print()
        print_success("Literature review generation completed!")
        console.print()

        # Save output
        save_output(result.query, result.generated_text, result.top_k_abstracts, output)

        # Display results
        display_results(result.generated_text, result.top_k_abstracts)

        # Final message
        console.print()
        print_success(f"Literature review has been generated and saved to: {output}")

    except ValidationError as e:
        print_error(f"Validation error: {str(e)}")
        raise typer.Exit(1)
    except ProcessingError as e:
        print_error(f"Processing error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        raise typer.Exit(1)


@app.command()
def run(
    output: Annotated[str, typer.Option("--output", "-o", help="Output file path")] = "generated_related_work.txt",
    persist_dir: Annotated[str, typer.Option("--persist-dir", "-p", help="ChromaDB persist directory")] = "./corpus-data/chroma_db",
    recreate_index: Annotated[bool, typer.Option("--recreate-index", help="Recreate index from scratch")] = False,
    hybrid_k: Annotated[int, typer.Option("--hybrid-k", help="Number of papers to retrieve")] = 50,
    num_score: Annotated[Optional[int], typer.Option("--num-score", help="Number of papers to score (None = all)")] = None,
    top_k: Annotated[int, typer.Option("--top-k", help="Number of top papers for related work")] = 3,
    relevance_model: Annotated[str, typer.Option("--relevance-model", help="Model for relevance scoring")] = "gpt-4o-mini",
    generation_model: Annotated[str, typer.Option("--generation-model", help="Model for text generation")] = "gpt-4o-mini",
):
    """
    Generate a "Related Work" section for a research paper using agentic RAG.

    This command performs the following steps:
    1. Prompt for CSV file path and research query
    2. Load and validate abstracts from CSV
    3. Initialize vector store with hybrid retrieval
    4. Retrieve relevant papers using hybrid search
    5. Score papers using relevance agent
    6. Select top-k most relevant papers
    7. Generate cohesive related work section
    8. Save and display results

    All other configuration can be customized via command-line flags.
    """

    print_section_header("AUTOMATED LITERATURE REVIEW GENERATION")

    # Interactive prompts for CSV path and research query
    console.print("\n[bold]Welcome! Let's generate your literature review.[/bold]\n")

    # Prompt for CSV path
    csv_path = prompt_for_csv_path()

    # Prompt for index recreation (override CLI flag with interactive choice)
    recreate_index = prompt_for_recreate_index()

    # Prompt for research query
    query = prompt_for_research_query()

    # Display query
    console.print(f"\n[bold]Research Query:[/bold]")
    console.print(Panel(query, style="dim"))

    # Create pipeline configuration
    config = PipelineConfig(
        persist_directory=persist_dir,
        recreate_index=recreate_index,
        hybrid_k=hybrid_k,
        num_abstracts_to_score=num_score,
        top_k=top_k,
        relevance_model=relevance_model,
        generation_model=generation_model,
    )

    # Configuration summary
    console.print(f"\nConfiguration:")
    console.print(f"  CSV file: {csv_path}")
    console.print(f"  Output file: {output}")
    console.print(f"  Persist directory: {config.persist_directory}")
    console.print(f"  Recreate index: {config.recreate_index}")
    console.print(f"  Hybrid retrieval k: {config.hybrid_k}")
    console.print(f"  Papers to score: {'All' if config.num_abstracts_to_score is None else config.num_abstracts_to_score}")
    console.print(f"  Top-k papers: {config.top_k}")
    console.print(f"  Relevance model: {config.relevance_model}")
    console.print(f"  Generation model: {config.generation_model}")

    # Execute pipeline
    try:
        # Initialize and run pipeline with spinner
        with console.status("[bold cyan]Running pipeline...[/bold cyan]", spinner="dots") as status:
            pipeline = LiteratureReviewPipeline(config)
            result = pipeline.run(csv_path=csv_path, query=query)

        console.print()
        print_success("Pipeline execution completed!")
        console.print()

        # Save output
        save_output(result.query, result.generated_text, result.top_k_abstracts, output)

        # Display results
        display_results(result.generated_text, result.top_k_abstracts)

        # Final message
        console.print()
        print_success(f"Related work has been generated and saved to: {output}")

    except ValidationError as e:
        print_error(f"Validation error: {str(e)}")
        raise typer.Exit(1)
    except ProcessingError as e:
        print_error(f"Processing error: {str(e)}")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        raise typer.Exit(1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
