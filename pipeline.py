"""
Literature Review Generation Pipeline

This module contains the core business logic for automated literature review generation.
All functions are pure (no side effects like printing) andP return results with metadata.
This makes them reusable in different contexts (CLI, notebooks, APIs, tests).
"""

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Annotated

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import openai
from agents import Agent, Runner
from dotenv import load_dotenv

from vectorstore import VectorStoreAbstract


# ============================================================================
# Models and Configuration
# ============================================================================

class AbstractRelevance(BaseModel):
    """Structured relevance assessment for a candidate paper."""
    id: int
    arguments_for: str
    arguments_for_quotes: list[str]
    arguments_against: str
    arguments_against_quotes: list[str]
    probability_score: Annotated[
        float,
        Field(ge=1.0, le=100.0, description="A relevance score between 1 and 100.")
    ]


@dataclass
class PipelineConfig:
    """Configuration for the literature review pipeline."""
    persist_directory: str = "./corpus-data/chroma_db"
    recreate_index: bool = False
    hybrid_k: int = 50
    num_abstracts_to_score: Optional[int] = None
    top_k: int = 3
    relevance_model: str = "gpt-4o-mini"
    generation_model: str = "gpt-4o-mini"
    random_seed: int = 42


@dataclass
class RetrievalStats:
    """Statistics from the retrieval phase."""
    total_papers_in_corpus: int
    papers_retrieved: int
    retrieval_rate: float
    retrieval_k: int


@dataclass
class ScoringStats:
    """Statistics from the relevance scoring phase."""
    papers_scored: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    median_score: float


@dataclass
class GenerationMetadata:
    """Metadata from the generation phase."""
    length_chars: int
    length_words: int
    total_citations: int
    unique_citations: int
    cited_paper_ids: List[int]


@dataclass
class IndexResult:
    """Result from the index building phase."""
    csv_path: str
    total_abstracts: int
    chunks_created: int
    total_indexed: int
    persist_directory: str
    recreated: bool
    config: PipelineConfig


@dataclass
class PipelineResult:
    """Complete result from the pipeline execution."""
    query: str
    generated_text: str
    top_k_abstracts: pd.DataFrame
    retrieval_stats: RetrievalStats
    scoring_stats: ScoringStats
    generation_metadata: GenerationMetadata
    all_abstracts: pd.DataFrame
    config: PipelineConfig


# ============================================================================
# Exception Classes
# ============================================================================

class ValidationError(Exception):
    """Raised when data validation fails."""
    pass


class ProcessingError(Exception):
    """Raised when processing fails."""
    pass


# ============================================================================
# Data Loading and Preparation
# ============================================================================

def validate_csv_path(csv_path: str) -> None:
    """Validate that CSV file exists and has correct extension."""
    path = Path(csv_path)

    if not path.exists():
        raise ValidationError(f"CSV file not found: {csv_path}")

    if not csv_path.lower().endswith('.csv'):
        raise ValidationError(f"File must be a CSV file (.csv): {csv_path}")


def load_abstracts_from_csv(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and validate abstracts from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (DataFrame, metadata dict)

    Raises:
        ValidationError: If file doesn't exist or validation fails
    """
    validate_csv_path(csv_path)

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValidationError(f"Failed to read CSV: {str(e)}")

    # Validate required columns
    required_columns = ['id', 'title', 'abstract']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValidationError(
            f"CSV missing required columns: {', '.join(missing_columns)}. "
            f"Required: {', '.join(required_columns)}. "
            f"Found: {', '.join(df.columns)}"
        )

    metadata = {
        'count': len(df),
        'columns': list(df.columns),
        'required_columns': required_columns
    }

    return df, metadata


def prepare_abstracts_for_indexing(
    df: pd.DataFrame,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Prepare abstracts for indexing by shuffling and combining title+abstract.

    Args:
        df: DataFrame with abstracts
        random_seed: Random seed for shuffling

    Returns:
        Tuple of (processed DataFrame, list of dicts for indexing)
    """
    # Shuffle for variety
    df_processed = df.sample(frac=1, random_state=random_seed).reset_index(drop=True).copy()

    # Concatenate title and abstract
    df_processed['title_abstract'] = df_processed['title'] + df_processed['abstract']

    # Convert to list of dictionaries
    samples_abstracts = [
        v for k, v in df_processed[['title_abstract', 'id']].reset_index(drop=True).T.to_dict().items()
    ]

    return df_processed, samples_abstracts


# ============================================================================
# Vector Store Operations
# ============================================================================

def initialize_vector_store(
    samples_abstracts: List[Dict[str, Any]],
    persist_directory: str,
    recreate_index: bool
) -> Tuple[VectorStoreAbstract, Dict[str, Any]]:
    """
    Initialize the vector store.

    Returns:
        Tuple of (VectorStoreAbstract, metadata dict)
    """
    try:
        vector_store = VectorStoreAbstract(
            abstracts=samples_abstracts,
            persist_directory=persist_directory,
            recreate_index=recreate_index
        )

        metadata = {
            'persist_directory': persist_directory,
            'index_exists': vector_store.index_exists,
            'doc_count': vector_store.get_document_count() if vector_store.index_exists else 0,
            'recreate_index': recreate_index
        }

        return vector_store, metadata

    except Exception as e:
        raise ProcessingError(f"Failed to initialize vector store: {str(e)}")


def process_and_index_documents(
    vector_store: VectorStoreAbstract
) -> Tuple[List, Dict[str, Any]]:
    """
    Chunk and index documents if needed.

    Returns:
        Tuple of (documents list, metadata dict)
    """
    if not vector_store.should_process_documents():
        return [], {
            'processed': False,
            'doc_count': vector_store.get_document_count(),
            'reason': 'using_existing_index'
        }

    # Chunk documents
    try:
        documents = vector_store.chunking()
    except Exception as e:
        raise ProcessingError(f"Failed to chunk documents: {str(e)}")

    # Index documents
    try:
        vector_store.index_document(documents)
    except Exception as e:
        raise ProcessingError(f"Failed to index documents: {str(e)}")

    return documents, {
        'processed': True,
        'chunks_created': len(documents),
        'total_indexed': vector_store.get_document_count()
    }


# ============================================================================
# Retrieval
# ============================================================================

def retrieve_relevant_papers(
    vector_store: VectorStoreAbstract,
    all_abstracts: pd.DataFrame,
    query: str,
    k: int
) -> Tuple[pd.DataFrame, RetrievalStats]:
    """
    Perform hybrid retrieval to find relevant papers.

    Returns:
        Tuple of (retrieved abstracts DataFrame, RetrievalStats)
    """
    try:
        rs = vector_store.hybrid_search(query, k=k)

        if rs is None:
            raise ProcessingError("Hybrid search returned no results")

        # Extract unique document IDs
        retrieved_docs = {item.metadata['id'] for item in rs}

        # Filter abstracts DataFrame
        retrieved_abstracts = all_abstracts[all_abstracts['id'].isin(retrieved_docs)].copy()

        stats = RetrievalStats(
            total_papers_in_corpus=len(all_abstracts),
            papers_retrieved=len(retrieved_abstracts),
            retrieval_rate=len(retrieved_abstracts) / len(all_abstracts) * 100,
            retrieval_k=k
        )

        return retrieved_abstracts, stats

    except Exception as e:
        raise ProcessingError(f"Failed to perform hybrid retrieval: {str(e)}")


# ============================================================================
# Relevance Scoring
# ============================================================================

def create_relevance_agent(model: str) -> Agent:
    """Create an agent that scores paper relevance using debate-style reasoning."""

    INSTRUCTIONS_DEBATE_RANKING = """
    You are a helpful research assistant who is helping with literature review of a research idea.
    You will be given a query or research idea and a candidate reference abstract.
    Your task is to score reference abstract based on their relevance to the query. Please make sure you read and understand these instructions carefully.
    Please keep this document open while reviewing, and refer to it as needed.

    ## Instruction:
    Use the following steps to rank the reference papers:

    1. Generate arguments for including this reference abstract in the literature review.

    2. Generate arguments against including this reference abstract in the literature review.

    3. Extract relevant sentences from the candidate paper abstract to support each argument.

    4. Then, provide a score between 1 and 100 (up to two decimal places) that is proportional to the probability
    of a paper with the given query including the candidate reference paper in its literature review.

    Important:
    - Put the extracted sentences in quotes
    - You can use the information in other candidate papers when generating the arguments for a candidate paper
    - Generate arguments and probability for each paper separately
    - Do not generate anything else apart from the probability and the arguments
    - Follow this process even if a candidate paper happens to be identical or near-perfect match to the query abstract

    Your Response: """

    relevance_agent = Agent(
        name="RelevanceAgent",
        instructions=INSTRUCTIONS_DEBATE_RANKING,
        model=model,
        output_type=AbstractRelevance
    )

    return relevance_agent


async def score_single_paper(
    id: int,
    query: str,
    reference_paper: str,
    model: str
) -> AbstractRelevance:
    """Score a single paper's relevance to the query."""
    relevance_agent = create_relevance_agent(model)

    user_instructions = f"""
For this query abstract with id={id}

Given the query abstract: {query}

Given the candidate reference paper abstract: {reference_paper}

Your Reference Abstract Relevance:
"""

    result = await Runner.run(relevance_agent, input=user_instructions)
    return result.final_output


async def score_papers_async(
    retrieved_abstracts: pd.DataFrame,
    query: str,
    model: str,
    num_to_score: Optional[int] = None
) -> List[AbstractRelevance]:
    """Score multiple abstracts in parallel."""

    # Select subset if specified
    abstracts_to_score = (
        retrieved_abstracts.head(num_to_score)
        if num_to_score is not None
        else retrieved_abstracts
    )

    # Create async tasks for parallel execution
    tasks = [
        asyncio.create_task(
            score_single_paper(
                id=item['id'],
                query=query,
                reference_paper=item['title_abstract'],
                model=model
            )
        )
        for index, item in abstracts_to_score[['id', 'title_abstract']].iterrows()
    ]

    results = await asyncio.gather(*tasks)
    return results


def score_papers_relevance(
    retrieved_abstracts: pd.DataFrame,
    query: str,
    relevance_model: str,
    num_to_score: Optional[int] = None
) -> Tuple[List[AbstractRelevance], ScoringStats]:
    """
    Score papers for relevance using the relevance agent.

    Returns:
        Tuple of (list of AbstractRelevance objects, ScoringStats)
    """
    try:
        # Handle async execution
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                results = loop.run_until_complete(
                    score_papers_async(retrieved_abstracts, query, relevance_model, num_to_score)
                )
            else:
                results = loop.run_until_complete(
                    score_papers_async(retrieved_abstracts, query, relevance_model, num_to_score)
                )
        except RuntimeError:
            results = asyncio.run(
                score_papers_async(retrieved_abstracts, query, relevance_model, num_to_score)
            )

        # Calculate statistics
        scores = [abs.probability_score for abs in results]

        stats = ScoringStats(
            papers_scored=len(scores),
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            median_score=float(np.median(scores))
        )

        return results, stats

    except Exception as e:
        raise ProcessingError(f"Failed to score abstracts: {str(e)}")


def select_top_papers(
    results: List[AbstractRelevance],
    retrieved_abstracts: pd.DataFrame,
    k: int
) -> pd.DataFrame:
    """
    Select top-k papers by relevance score.

    Returns:
        DataFrame with top-k papers and relevance scores
    """
    try:
        # Get top-k scores
        scores = [(abs.id, abs.probability_score) for abs in results]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_k_scores = sorted_scores[:k]

        # Extract IDs and get full information
        top_k_id = [id for id, score in top_k_scores]
        top_k_abstracts = retrieved_abstracts[retrieved_abstracts['id'].isin(top_k_id)].copy()

        # Add scores
        score_dict = {id: score for id, score in top_k_scores}
        top_k_abstracts['relevance_score'] = top_k_abstracts['id'].map(score_dict)
        top_k_abstracts = top_k_abstracts.sort_values('relevance_score', ascending=False)

        return top_k_abstracts

    except Exception as e:
        raise ProcessingError(f"Failed to select top-k papers: {str(e)}")


# ============================================================================
# Text Generation
# ============================================================================

def generate_related_work_text(
    query: str,
    top_k_abstracts: pd.DataFrame,
    generation_model: str
) -> Tuple[str, GenerationMetadata]:
    """
    Generate related work section.

    Returns:
        Tuple of (generated text, GenerationMetadata)
    """
    INSTRUCTIONS_RELATED_WORK = """
    You are an expert research assistant who is helping with literature review for a research idea or abstract.
    You will be provided with an abstract or research idea and a list of reference abstracts.
    Your task is to write the related work section of the document using only the provided reference abstracts.
    Please write the related work section creating a cohesive storyline by doing a critical analysis of prior work
    in the reference abstracts comparing the strengths and weaknesses while also motivating the proposed approach.
    You should cite the reference abstracts as [id] whenever you are referring it in the related work.
    Do not write it as Reference #. Do not cite abstract or research Idea.
    Do not include any extra notes or newline characters at the end.
    Do not copy the abstracts of reference papers directly but compare and contrast to the main work concisely.
    Do not provide the output in bullet points or markdown.
    Do not provide references at the end.
    Please cite all the provided reference papers if needed.
    """

    try:
        # Build input
        input_related_work = f"Given the Research Idea or abstract: {query}"
        input_related_work += "\n\n## Given references abstracts list below:"

        for index, item in top_k_abstracts[['id', 'title_abstract']].iterrows():
            input_related_work += f"\n\n[{item['id']}]: {item['title_abstract']}"

        input_related_work += "\n\nWrite the related work section summarizing in a cohesive story prior works relevant to the research idea."
        input_related_work += "\n\n## Related Work:"

        # Generate
        openai_client = openai.OpenAI()
        response = openai_client.responses.create(
            model=generation_model,
            instructions=INSTRUCTIONS_RELATED_WORK,
            input=input_related_work
        )

        generated_text = response.output_text

        # Extract citation information
        citations = re.findall(r'\[(\d+)\]', generated_text)
        unique_citations = sorted(set(int(c) for c in citations))

        metadata = GenerationMetadata(
            length_chars=len(generated_text),
            length_words=len(generated_text.split()),
            total_citations=len(citations),
            unique_citations=len(unique_citations),
            cited_paper_ids=unique_citations
        )

        return generated_text, metadata

    except Exception as e:
        raise ProcessingError(f"Failed to generate related work: {str(e)}")


# ============================================================================
# Output Formatting
# ============================================================================

def format_output_for_file(
    query: str,
    generated_text: str,
    top_k_abstracts: pd.DataFrame
) -> str:
    """
    Format the complete output for saving to file.

    Returns:
        Formatted string ready to write to file
    """
    citations = re.findall(r'\[(\d+)\]', generated_text)
    unique_citations = sorted(set(int(c) for c in citations))

    output = []
    output.append("=" * 80)
    output.append("AUTOMATED LITERATURE REVIEW GENERATION")
    output.append("=" * 80 + "\n")

    output.append("RESEARCH QUERY:")
    output.append("-" * 80)
    output.append(query)
    output.append("\n" + "=" * 80 + "\n")

    output.append("RELATED WORK:")
    output.append("-" * 80)
    output.append(generated_text)
    output.append("\n" + "=" * 80 + "\n")

    output.append("REFERENCES:")
    output.append("-" * 80)
    for paper_id in unique_citations:
        paper = top_k_abstracts[top_k_abstracts['id'] == paper_id]
        if not paper.empty:
            output.append(f"[{paper_id}] {paper.iloc[0]['title']}")
            output.append(f"    {paper.iloc[0]['abstract'][:200]}...\n")

    output.append("=" * 80)

    return "\n".join(output)


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

class LiteratureReviewPipeline:
    """
    Main orchestrator for the literature review generation pipeline.

    This class provides a high-level interface to run the entire pipeline
    without any UI concerns. Perfect for use in notebooks, APIs, or testing.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration."""
        self.config = config
        self.vector_store = None
        self.all_abstracts = None

        # Load environment
        load_dotenv(override=True)

    def build_index(self, csv_path: str) -> IndexResult:
        """
        Build or update the vector store index from CSV data.

        This method performs Steps 1-3 of the pipeline:
        1. Load and prepare data from CSV
        2. Initialize vector store
        3. Process and index documents

        Args:
            csv_path: Path to CSV file containing abstracts

        Returns:
            IndexResult with indexing statistics and metadata

        Raises:
            ValidationError: If inputs are invalid
            ProcessingError: If processing fails
        """
        # Step 1: Load and prepare data
        self.all_abstracts, _ = load_abstracts_from_csv(csv_path)
        self.all_abstracts, samples_abstracts = prepare_abstracts_for_indexing(
            self.all_abstracts,
            self.config.random_seed
        )

        # Step 2: Initialize vector store
        self.vector_store, _ = initialize_vector_store(
            samples_abstracts,
            self.config.persist_directory,
            self.config.recreate_index
        )

        # Step 3: Process and index documents
        documents, index_metadata = process_and_index_documents(self.vector_store)

        # Build result
        return IndexResult(
            csv_path=csv_path,
            total_abstracts=len(self.all_abstracts),
            chunks_created=len(documents) if documents else 0,
            total_indexed=self.vector_store.get_document_count(),
            persist_directory=self.config.persist_directory,
            recreated=self.config.recreate_index,
            config=self.config
        )

    def load_abstracts_only(self, csv_path: str) -> Dict[str, Any]:
        """
        Load abstracts from CSV without creating or updating the index.

        This is a lightweight method for loading abstract metadata when you
        want to generate reviews from an existing index without re-indexing.

        Args:
            csv_path: Path to CSV file containing abstracts

        Returns:
            Dict with metadata about loaded abstracts

        Raises:
            ValidationError: If CSV is invalid
        """
        # Load and prepare abstracts (same as build_index steps 1)
        self.all_abstracts, _ = load_abstracts_from_csv(csv_path)
        self.all_abstracts, _ = prepare_abstracts_for_indexing(
            self.all_abstracts,
            self.config.random_seed
        )

        return {
            'csv_path': csv_path,
            'total_abstracts': len(self.all_abstracts),
            'columns': list(self.all_abstracts.columns)
        }

    def generate_review(self, query: str) -> PipelineResult:
        """
        Generate a literature review from an existing index.

        This method performs Steps 4-7 of the pipeline:
        4. Retrieve relevant papers using hybrid search
        5. Score papers for relevance
        6. Select top-k papers
        7. Generate related work text

        Requires that build_index() has been called first, or that the
        vector store already exists at the configured persist_directory.

        Args:
            query: Research query/abstract

        Returns:
            PipelineResult with all outputs and metadata

        Raises:
            ProcessingError: If vector store not initialized or generation fails
        """
        # Ensure vector store is initialized
        if self.vector_store is None:
            # Try to load existing index
            try:
                samples_abstracts = []  # Empty list since we're loading existing
                self.vector_store, _ = initialize_vector_store(
                    samples_abstracts,
                    self.config.persist_directory,
                    recreate_index=False  # Never recreate when generating
                )

                if not self.vector_store.index_exists:
                    raise ProcessingError(
                        f"No index found at {self.config.persist_directory}. "
                        "Please run build_index() first or use the 'index' command."
                    )
            except Exception as e:
                raise ProcessingError(
                    f"Failed to load vector store: {str(e)}. "
                    "Please run build_index() first or use the 'index' command."
                )

        # Load abstracts if not already loaded
        if self.all_abstracts is None:
            raise ProcessingError(
                "Abstracts not loaded. Please run load_abstracts_only() or build_index() first with the CSV path."
            )

        # Step 4: Retrieve relevant papers
        retrieved_abstracts, retrieval_stats = retrieve_relevant_papers(
            self.vector_store,
            self.all_abstracts,
            query,
            self.config.hybrid_k
        )

        # Step 5: Score papers for relevance
        relevance_results, scoring_stats = score_papers_relevance(
            retrieved_abstracts,
            query,
            self.config.relevance_model,
            self.config.num_abstracts_to_score
        )

        # Step 6: Select top-k papers
        top_k_abstracts = select_top_papers(
            relevance_results,
            retrieved_abstracts,
            self.config.top_k
        )

        # Step 7: Generate related work text
        generated_text, generation_metadata = generate_related_work_text(
            query,
            top_k_abstracts,
            self.config.generation_model
        )

        # Return complete result
        return PipelineResult(
            query=query,
            generated_text=generated_text,
            top_k_abstracts=top_k_abstracts,
            retrieval_stats=retrieval_stats,
            scoring_stats=scoring_stats,
            generation_metadata=generation_metadata,
            all_abstracts=self.all_abstracts,
            config=self.config
        )

    def run(self, csv_path: str, query: str) -> PipelineResult:
        """
        Execute the complete literature review generation pipeline.

        This is a convenience method that combines build_index() and generate_review()
        for backward compatibility and ease of use.

        Args:
            csv_path: Path to CSV file containing abstracts
            query: Research query/abstract

        Returns:
            PipelineResult with all outputs and metadata

        Raises:
            ValidationError: If inputs are invalid
            ProcessingError: If processing fails
        """
        # Build index (Steps 1-3)
        self.build_index(csv_path)

        # Generate review (Steps 4-7)
        return self.generate_review(query)
