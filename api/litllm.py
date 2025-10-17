"""
LitLLM REST API

FastAPI-based REST API for the Agentic Literature Review Generation pipeline.
Provides HTTP endpoints to generate literature reviews programmatically.
"""

import os
import sys
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

# Add parent directory to path to import pipeline
sys.path.append(str(Path(__file__).parent.parent))

from pipeline import (
    LiteratureReviewPipeline,
    PipelineConfig,
    PipelineResult,
    ValidationError,
    ProcessingError,
)

# ============================================================================
# FastAPI Application Setup
# ============================================================================

app = FastAPI(
    title="LitLLM API",
    description="REST API for Agentic Literature Review Generation using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware - Allow all origins for development
# In production, restrict to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models (Request/Response)
# ============================================================================

class ConfigOverrides(BaseModel):
    """Optional configuration overrides for the pipeline."""
    persist_directory: Optional[str] = None
    recreate_index: Optional[bool] = None
    hybrid_k: Optional[int] = Field(None, ge=1, le=1000)
    num_abstracts_to_score: Optional[int] = Field(None, ge=1)
    top_k: Optional[int] = Field(None, ge=1, le=50)
    relevance_model: Optional[str] = None
    generation_model: Optional[str] = None
    random_seed: Optional[int] = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hybrid_k": 50,
                "top_k": 5,
                "relevance_model": "gpt-4o-mini"
            }
        }
    )


class LiteratureReviewRequest(BaseModel):
    """Request model for literature review generation."""
    query: str = Field(..., min_length=10, description="Research query or idea")
    csv_path: str = Field(..., description="Path to CSV file with abstracts")
    config: Optional[ConfigOverrides] = Field(None, description="Optional configuration overrides")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Deep learning approaches for natural language processing",
                "csv_path": "./abstracts_rag.csv",
                "config": {
                    "hybrid_k": 50,
                    "top_k": 3
                }
            }
        }
    )


class PaperSummary(BaseModel):
    """Simplified paper representation."""
    id: int
    title: str
    abstract: str
    relevance_score: float


class RetrievalStatsResponse(BaseModel):
    """Retrieval statistics."""
    total_papers_in_corpus: int
    papers_retrieved: int
    retrieval_rate: float
    retrieval_k: int


class ScoringStatsResponse(BaseModel):
    """Scoring statistics."""
    papers_scored: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    median_score: float


class GenerationMetadataResponse(BaseModel):
    """Generation metadata."""
    length_chars: int
    length_words: int
    total_citations: int
    unique_citations: int
    cited_paper_ids: List[int]


class LiteratureReviewResponse(BaseModel):
    """Complete response from literature review generation."""
    query: str
    generated_text: str
    top_papers: List[PaperSummary]
    retrieval_stats: RetrievalStatsResponse
    scoring_stats: ScoringStatsResponse
    generation_metadata: GenerationMetadataResponse
    total_corpus_size: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "Deep learning for NLP",
                "generated_text": "The field of natural language processing has...",
                "top_papers": [
                    {
                        "id": 123,
                        "title": "Attention Is All You Need",
                        "abstract": "The dominant sequence transduction models...",
                        "relevance_score": 95.5
                    }
                ],
                "retrieval_stats": {
                    "total_papers_in_corpus": 1000,
                    "papers_retrieved": 50,
                    "retrieval_rate": 5.0,
                    "retrieval_k": 50
                },
                "scoring_stats": {
                    "papers_scored": 50,
                    "mean_score": 65.3,
                    "std_score": 12.4,
                    "min_score": 35.2,
                    "max_score": 95.5,
                    "median_score": 67.1
                },
                "generation_metadata": {
                    "length_chars": 1250,
                    "length_words": 215,
                    "total_citations": 5,
                    "unique_citations": 3,
                    "cited_paper_ids": [123, 456, 789]
                },
                "total_corpus_size": 1000
            }
        }
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    message: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    status_code: int


# ============================================================================
# Helper Functions
# ============================================================================

def convert_dataframe_to_papers(df: pd.DataFrame) -> List[PaperSummary]:
    """Convert DataFrame to list of PaperSummary objects."""
    papers = []
    for _, row in df.iterrows():
        papers.append(PaperSummary(
            id=int(row['id']),
            title=str(row['title']),
            abstract=str(row['abstract']),
            relevance_score=float(row['relevance_score'])
        ))
    return papers


def cleanup_temp_file(filepath: str):
    """Remove temporary file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Warning: Could not delete temp file {filepath}: {e}")


def build_pipeline_config(overrides: Optional[ConfigOverrides] = None) -> PipelineConfig:
    """Build PipelineConfig with optional overrides."""
    # Start with defaults
    config_dict = {
        "persist_directory": "./corpus-data/chroma_db",
        "recreate_index": False,
        "hybrid_k": 50,
        "num_abstracts_to_score": None,
        "top_k": 3,
        "relevance_model": "gpt-4o-mini",
        "generation_model": "gpt-4o-mini",
        "random_seed": 42
    }

    # Apply overrides
    if overrides:
        for key, value in overrides.dict(exclude_none=True).items():
            config_dict[key] = value

    return PipelineConfig(**config_dict)


async def run_pipeline_wrapper(
    query: str,
    csv_path: str,
    config: PipelineConfig
) -> LiteratureReviewResponse:
    """
    Wrapper to run the pipeline and convert result to API response.

    Uses ThreadPoolExecutor to run the blocking pipeline in a separate thread,
    avoiding event loop conflicts with uvloop/asyncio.

    Raises:
        HTTPException: On validation or processing errors
    """
    try:
        # Run blocking pipeline in thread pool to avoid event loop conflicts
        # This is necessary because:
        # 1. FastAPI runs with uvloop (high-performance event loop)
        # 2. pipeline.run() is synchronous but calls async code internally
        # 3. nest_asyncio cannot patch uvloop
        # Solution: Run in separate thread to isolate from event loop
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        def run_pipeline():
            """Helper function to run pipeline in thread."""
            pipeline = LiteratureReviewPipeline(config)
            return pipeline.run(csv_path=csv_path, query=query)

        # Execute pipeline in thread pool
        result: PipelineResult = await loop.run_in_executor(executor, run_pipeline)

        # Convert DataFrame to list of PaperSummary
        top_papers = convert_dataframe_to_papers(result.top_k_abstracts)

        # Build response
        response = LiteratureReviewResponse(
            query=result.query,
            generated_text=result.generated_text,
            top_papers=top_papers,
            retrieval_stats=RetrievalStatsResponse(
                total_papers_in_corpus=result.retrieval_stats.total_papers_in_corpus,
                papers_retrieved=result.retrieval_stats.papers_retrieved,
                retrieval_rate=result.retrieval_stats.retrieval_rate,
                retrieval_k=result.retrieval_stats.retrieval_k
            ),
            scoring_stats=ScoringStatsResponse(
                papers_scored=result.scoring_stats.papers_scored,
                mean_score=result.scoring_stats.mean_score,
                std_score=result.scoring_stats.std_score,
                min_score=result.scoring_stats.min_score,
                max_score=result.scoring_stats.max_score,
                median_score=result.scoring_stats.median_score
            ),
            generation_metadata=GenerationMetadataResponse(
                length_chars=result.generation_metadata.length_chars,
                length_words=result.generation_metadata.length_words,
                total_citations=result.generation_metadata.total_citations,
                unique_citations=result.generation_metadata.unique_citations,
                cited_paper_ids=result.generation_metadata.cited_paper_ids
            ),
            total_corpus_size=len(result.all_abstracts)
        )

        return response

    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except ProcessingError as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"File not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information."""
    return {
        "name": "LitLLM API",
        "version": "1.0.0",
        "description": "REST API for Agentic Literature Review Generation",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse: API health status
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        message="LitLLM API is running"
    )


@app.get("/api/v1/config/defaults", tags=["Configuration"])
async def get_default_config():
    """
    Get default pipeline configuration.

    Returns:
        dict: Default configuration values
    """
    default_config = PipelineConfig()
    return {
        "persist_directory": default_config.persist_directory,
        "recreate_index": default_config.recreate_index,
        "hybrid_k": default_config.hybrid_k,
        "num_abstracts_to_score": default_config.num_abstracts_to_score,
        "top_k": default_config.top_k,
        "relevance_model": default_config.relevance_model,
        "generation_model": default_config.generation_model,
        "random_seed": default_config.random_seed
    }


@app.post("/api/v1/literature-review", response_model=LiteratureReviewResponse, tags=["Literature Review"])
async def generate_literature_review(request: LiteratureReviewRequest):
    """
    Generate a literature review from a CSV file.

    Args:
        request: LiteratureReviewRequest with query, csv_path, and optional config

    Returns:
        LiteratureReviewResponse: Complete literature review with metadata

    Raises:
        HTTPException: On validation or processing errors
    """
    # Build configuration
    config = build_pipeline_config(request.config)

    # Run pipeline
    response = await run_pipeline_wrapper(
        query=request.query,
        csv_path=request.csv_path,
        config=config
    )

    return response


@app.post("/api/v1/literature-review/upload", response_model=LiteratureReviewResponse, tags=["Literature Review"])
async def generate_literature_review_with_upload(
    background_tasks: BackgroundTasks,
    query: str = Form(..., description="Research query or idea"),
    csv_file: UploadFile = File(..., description="CSV file with abstracts"),
    hybrid_k: int = Form(50, description="Number of papers to retrieve"),
    top_k: int = Form(3, description="Number of top papers for final review"),
    num_abstracts_to_score: Optional[int] = Form(None, description="Number of papers to score (None = all)"),
    relevance_model: str = Form("gpt-4o-mini", description="Model for relevance scoring"),
    generation_model: str = Form("gpt-4o-mini", description="Model for text generation")
):
    """
    Generate a literature review from an uploaded CSV file.

    Args:
        background_tasks: FastAPI background tasks for cleanup
        query: Research query or idea
        csv_file: Uploaded CSV file
        hybrid_k: Number of papers to retrieve
        top_k: Number of top papers for final review
        num_abstracts_to_score: Number of papers to score (None = all)
        relevance_model: Model for relevance scoring
        generation_model: Model for text generation

    Returns:
        LiteratureReviewResponse: Complete literature review with metadata

    Raises:
        HTTPException: On validation or processing errors
    """
    # Validate file type
    if not csv_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file (.csv)")

    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    temp_path = temp_file.name

    try:
        # Write uploaded file to temp location
        contents = await csv_file.read()
        temp_file.write(contents)
        temp_file.close()

        # Build configuration
        config_overrides = ConfigOverrides(
            hybrid_k=hybrid_k,
            top_k=top_k,
            num_abstracts_to_score=num_abstracts_to_score,
            relevance_model=relevance_model,
            generation_model=generation_model
        )
        config = build_pipeline_config(config_overrides)

        # Run pipeline
        response = await run_pipeline_wrapper(
            query=query,
            csv_path=temp_path,
            config=config
        )

        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_path)

        return response

    except Exception as e:
        # Clean up temp file on error
        cleanup_temp_file(temp_path)
        raise


# ============================================================================
# Main Entry Point (for development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    print("Starting LitLLM API server...")
    print("Docs available at: http://localhost:8000/docs")

    uvicorn.run(
        "litllm:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
