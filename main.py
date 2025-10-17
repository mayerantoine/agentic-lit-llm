#!/usr/bin/env python3
"""
Automated Literature Review Generation CLI

This CLI application automates the generation of "Related Work" sections for scientific papers
using an agentic RAG (Retrieval-Augmented Generation) pipeline.
"""

import os

# Set tokenizers parallelism to false to avoid fork warnings
# This must be set before importing any libraries that use HuggingFace tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
from pathlib import Path
from typing import Optional, Annotated

import pandas as pd
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel

from pipeline import (
    PipelineConfig,
    PipelineResult,
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
        # Extract citations
        citations = re.findall(r'\[(\d+)\]', generated_text)
        unique_citations = sorted(set(int(c) for c in citations))

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

    # Extract and display citations
    citations = re.findall(r'\[(\d+)\]', generated_text)
    unique_citations = sorted(set(int(c) for c in citations))

    console.print(f"\nCited Papers:")
    for paper_id in unique_citations:
        paper = top_k_abstracts[top_k_abstracts['id'] == paper_id]
        if not paper.empty:
            console.print(f"  [{paper_id}] {paper.iloc[0]['title']}")


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
