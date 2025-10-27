import pandas as pd
import os
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

class VectorStoreAbstract():
    def __init__(self, abstracts: List = None, persist_directory: str = "", recreate_index: bool = True):
        self.abstracts = abstracts
        self.persist_directory = persist_directory
        self.recreate_index = recreate_index
        
        # Initialize components
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[Any] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self.hybrid_retriever = None
        self._cached_documents: Optional[List[Document]] = None

        self.initialize_store()
    
    def initialize_store(self):
        persist_path = Path(self.persist_directory)
        persist_path.mkdir(parents=True, exist_ok=True)

        # Check if index already exists
        self.index_exists = self._check_index_exists()

        if not self.recreate_index and self.index_exists:
            print(f"Using existing index at {self.persist_directory}")
            print(f"   Set recreate_index=True to force recreation")
        else:
            if self.index_exists and self.recreate_index:
                print(f"Recreating existing index at {self.persist_directory}")
            else:
                print(f"Creating new index at {self.persist_directory}")


        self.text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=150,
                        chunk_overlap=20,
                        length_function=len,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""])

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            show_progress=False)


        # Initialize ChromaDB vector store
        self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )

        # Delete existing collection if recreate_index=True
        if self.recreate_index and self.index_exists:
            try:
                print(f"   Deleting existing collection...")
                # Get all collection names and delete them
                # ChromaDB creates collections, we need to delete and recreate
                collection_name = self.vectorstore._collection.name
                self.vectorstore._client.delete_collection(name=collection_name)
                print(f"   Collection '{collection_name}' deleted successfully")

                # Reinitialize vectorstore with empty collection
                self.vectorstore = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=self.persist_directory
                )
                print(f"   New empty collection created")
            except Exception as e:
                print(f"   Warning: Could not delete existing collection: {e}")
                print(f"   Continuing with existing collection...")
    
    def create_hybrid_retriever(self,documents, weights=[0.5, 0.5], k=20):
        """
        Creates hybrid retriever combining semantic + keyword search
        weights: [semantic_weight, keyword_weight]
        """
        # Semantic retriever
        semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Keyword retriever (BM25)
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        
        # Combine them
        ensemble_retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=weights
        )
        
        return ensemble_retriever

    def chunking(self):
        all_chunked_documents: List[Document] = []

        # If no html_articles provided, return empty list (index already exists)
        if self.abstracts is None:
            return all_chunked_documents

        total_articles = len(self.abstracts)

        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_articles, desc="Chunking documents", unit="article")
        except ImportError:
            progress_bar = None
            print("Processing articles (this may take several minutes)...")

        try:
            for content in self.abstracts:

                # Second pass: Apply RecursiveCharacterTextSplitter for size constraints
                size_constrained_chunks = self.text_splitter.split_text(content['title_abstract'])
                for i, chunk in enumerate(size_constrained_chunks):
                    chunked_document = Document(
                                    page_content=chunk,
                                    metadata={ "id":content['id']}
                                )
                    all_chunked_documents.append(chunked_document)

                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)

        finally:
            if progress_bar:
                progress_bar.close()

        return all_chunked_documents
    
    def index_document(self,all_chunked_documents,batch_size=50):

        total_docs = len(all_chunked_documents)
        processed_count = 0

        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=total_docs, desc="Creating embeddings", unit="doc")
        except ImportError:
            progress_bar = None
            print("   Processing in batches (this may take several minutes)...")

        try:
            # Process documents in batches
            for i in range(0, len(all_chunked_documents), batch_size):
                batch = all_chunked_documents[i:i + batch_size]

                # Add remaining batches to existing vectorstore
                if self.vectorstore is not None:
                    self.vectorstore.add_documents(batch)

                processed_count += len(batch)

                # Update progress bar
                if progress_bar:
                    progress_bar.update(len(batch))
                else:
                    print(f"   Processed {processed_count}/{total_docs} documents")
        finally:
            if progress_bar:
                progress_bar.close()

        # Create hybrid retriever with newly indexed documents
        self.hybrid_retriever = self.create_hybrid_retriever(documents = all_chunked_documents)
        # Store documents for later use
        self._cached_documents = all_chunked_documents


    def semantic_search(self,
            query: str,
            k: int = 5,
            search_type: str = "mmr"
        ) -> List[Dict[str, Any]]:
            """
            Perform semantic search on indexed documents.
            
            Args:
                query: Search query string
                k: Number of results to return
                collection_filter: Optional collection filter ('pcd', 'eid', 'mmwr')
                search_type: Type of search ('similarity', 'mmr')
                
            Returns:
                List of search results with content and metadata
            """

            
            # Update retriever with search parameters
            retriever = self.vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k":k}
            )
            
            # Perform search
            docs = retriever.invoke(query)
            
            # Format results
            results: List[Dict[str, Any]] = []
            for doc in docs:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                results.append(result)
            
            return results




    def _ensure_hybrid_retriever(self):
        """Ensure hybrid retriever is initialized, creating it if necessary."""
        if self.hybrid_retriever is not None:
            return  # Already initialized

        # If we have cached documents (from recent indexing), use them
        if self._cached_documents:
            self.hybrid_retriever = self.create_hybrid_retriever(documents=self._cached_documents)
            return

        # Otherwise, need to load documents from vectorstore
        # Get all documents from vectorstore
        if self.vectorstore and self.index_exists:
            print("Loading documents from existing index for hybrid retrieval...")
            try:
                # Get all documents from vectorstore
                collection = self.vectorstore._collection
                results = collection.get(include=['documents', 'metadatas'])

                # Reconstruct Document objects
                documents = []
                if results and 'documents' in results:
                    for i, doc_content in enumerate(results['documents']):
                        metadata = results['metadatas'][i] if 'metadatas' in results else {}
                        documents.append(Document(page_content=doc_content, metadata=metadata))

                if documents:
                    self.hybrid_retriever = self.create_hybrid_retriever(documents=documents)
                    self._cached_documents = documents
                    print(f"Hybrid retriever initialized with {len(documents)} documents")
                else:
                    print("Warning: No documents found in vectorstore")
            except Exception as e:
                print(f"Warning: Could not initialize hybrid retriever: {e}")
                print("Falling back to semantic search only")

    def hybrid_search(self,query: str, k: int = 20):
        """Hybrid search using ensemble retriever"""
        # Ensure hybrid retriever is initialized
        self._ensure_hybrid_retriever()

        if self.hybrid_retriever:
            results = self.hybrid_retriever.invoke(query)
            return results[:k]
        else:
            print("Warning: Hybrid retriever not available, returning None")
            return None
    
    def _check_index_exists(self) -> bool:
        """Check if a ChromaDB index already exists in the persist directory."""
        persist_path = Path(self.persist_directory)
        
        # Check for ChromaDB files
        chroma_files = [
            persist_path / "chroma.sqlite3",
        ]
        
        # Check if any essential ChromaDB files exist
        has_db_files = any(file.exists() for file in chroma_files)
        
        # Also check for collection directories (ChromaDB creates UUID-named folders)
        has_collections = False
        if persist_path.exists():
            for item in persist_path.iterdir():
                if item.is_dir() and len(item.name) == 36:  # UUID length
                    has_collections = True
                    break
        
        return has_db_files and has_collections

    def should_process_documents(self) -> bool:
        """Determine if documents should be processed (chunked and indexed)."""
        return self.recreate_index or not self.index_exists


    def get_document_count(self) -> int:
        """Get the number of documents in the existing index."""
        if self.vectorstore and self.index_exists:
            try:
                # Try to get collection info
                collection = self.vectorstore._collection
                return collection.count()
            except Exception:
                return 0
        return 0

