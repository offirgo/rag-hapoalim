import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from enum import Enum

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.embeddings.sentence_transformers_embedder import SentenceTransformersEmbedder
from src.vector_stores.faiss_store import FaissVectorStore
from src.llm.echo import EchoLLM
from src.llm.ollama_llm import OllamaLLM
from src.llm.base import LLMError
from src.llm.rag_service import RAGService
from src.document_processor.word_processor import WordProcessor
from src.document_processor.excel_processor import ExcelProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class LLMType(str, Enum):
    ECHO = "echo"
    TINY_LLAMA = "tinyllama"

# Define models for API requests/responses
class QuestionRequest(BaseModel):
    question: str = Field(..., description="User's question")
    top_k: Optional[int] = Field(3, description="Number of chunks to retrieve")
    min_score: Optional[float] = Field(0.5, description="Minimum similarity score")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")
    llm_type: Optional[str] = Field("tinyllama", description="LLM to use: 'echo' or 'tinyllama'")



class AnswerResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    retrieval_time: float = Field(..., description="Time spent on retrieval")
    generation_time: float = Field(..., description="Time spent on generation")
    total_time: float = Field(..., description="Total processing time")


class SystemInfoResponse(BaseModel):
    vector_store: Dict[str, Any] = Field(..., description="Vector store information")
    llm: Dict[str, Any] = Field(..., description="LLM information")
    config: Dict[str, Any] = Field(..., description="System configuration")


# Initialize environment variables with defaults
PROJECT_ROOT = Path(__file__).parent.parent.parent
INDEX_PATH = os.environ.get("RAG_INDEX_PATH", str(PROJECT_ROOT / "vector_index"))
DATA_DIR = os.environ.get("RAG_DATA_DIR", str(PROJECT_ROOT / "data"))

LLM_TYPE = os.environ.get("RAG_LLM_TYPE", "echo")  # 'echo' or 'ollama'
OLLAMA_MODEL = os.environ.get("RAG_OLLAMA_MODEL", "tinyllama")

# Create upload directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for documents",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
embedder = SentenceTransformersEmbedder()
vector_store = FaissVectorStore(embedding_dimension=embedder.embedding_dim, embedder=embedder)
word_processor = WordProcessor()
excel_processor = ExcelProcessor()

# Load vector index if it exists
index_path = Path(INDEX_PATH)
if index_path.exists():
    logger.info(f"Loading vector index from {index_path}...")
    vector_store.load_index(index_path)
    logger.info(f"Loaded {vector_store.get_chunk_count()} chunks")
else:
    logger.warning(f"Vector index not found at {index_path}. Starting with empty index.")

# Initialize LLM
if LLM_TYPE.lower() == "ollama":
    try:
        llm = OllamaLLM(model=OLLAMA_MODEL)
        logger.info(f"Using Ollama LLM with model {OLLAMA_MODEL}")
    except LLMError as e:
        logger.error(f"Failed to initialize Ollama LLM: {e}. Falling back to EchoLLM.")
        llm = EchoLLM()
else:
    llm = EchoLLM()
    logger.info("Using EchoLLM for testing")

# Initialize RAG service
rag_service = RAGService(
    vector_store=vector_store,
    llm=llm,
    config={
        "default_top_k": 3,
        "default_min_score": 0.5
    }
)
def get_llm(llm_type: str):
    """Get an LLM instance based on the specified type"""
    if llm_type.lower() == "echo":
        return EchoLLM()
    elif llm_type.lower() == "tinyllama":
        try:
            return OllamaLLM(model="tinyllama")
        except LLMError as e:
            logger.error(f"Failed to initialize Ollama LLM: {e}. Falling back to EchoLLM.")
            return EchoLLM()
    else:
        logger.warning(f"Unknown LLM type: {llm_type}. Using tinyllama.")
        return OllamaLLM(model="tinyllama")

# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint returning basic API information"""
    return {
        "name": "RAG API",
        "version": "0.1.0",
        "status": "active",
        "docs": "/docs"
    }


@app.get("/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get information about the RAG system"""
    return rag_service.get_system_info()


@app.post("/answer", response_model=AnswerResponse,
          description="""
          **LLM Options:**
        - **echo**: Testing mode that doesn't use a real AI model. Returns a placeholder response showing what documents were retrieved.
        - **tinyllama**: Uses the TinyLlama AI model to generate a real answer based on the retrieved documents.
            """)
async def answer_question(request: QuestionRequest):
    """Answer a question using RAG"""
    try:
        # Get the appropriate LLM based on the request
        llm = get_llm(request.llm_type)

        # Create a RAG service with this LLM
        temp_rag_service = RAGService(
            vector_store=vector_store,
            llm=llm,
            config={
                "default_top_k": 3,
                "default_min_score": 0.5
            }
        )

        # Generate answer
        result = temp_rag_service.answer(
            query=request.question,
            top_k=request.top_k,
            min_score=request.min_score,
            filters=request.filters
        )

        return result
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/upload", response_model=Dict[str, Any])
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Log file details to help with debugging
        logger.info(
            f"Received upload request for file: {file.filename}, content_type: {file.content_type}, size: {file.size}")

        # Validate file type
        if not file.filename.lower().endswith((".docx", ".xlsx", ".xls")):
            logger.warning(f"Unsupported file type: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Only .docx, .xlsx, and .xls are supported"
            )

        # Save uploaded file to the data directory
        file_path = Path(DATA_DIR) / file.filename

        # Read file in chunks to handle large files
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file received")

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File saved to {file_path}")

        # Process based on file type
        if file.filename.lower().endswith(".docx"):
            document = word_processor.process(file_path)
        else:  # Excel file
            document = excel_processor.process(file_path)

        # Generate embeddings
        document_with_embeddings = document.with_embeddings(embedder)

        # Add to vector store
        vector_store.add_document(document_with_embeddings)

        # Save updated index
        vector_store.save_index(INDEX_PATH)

        logger.info(f"Successfully processed file {file.filename}, added {len(document.chunks)} chunks")

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_added": len(document.chunks),
            "total_chunks": vector_store.get_chunk_count()
        }

    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/files", response_model=List[Dict[str, Any]])
async def list_files():
    """List all files and their indexing status"""
    try:
        # Get all indexed sources from the vector store
        indexed_sources = set()
        for meta in vector_store._chunk_metadata.values():
            if 'source' in meta:
                source = meta['source']
                indexed_sources.add(source)

        logger.info(f"Found {len(indexed_sources)} indexed sources: {indexed_sources}")

        # Get files from data directory
        data_path = Path(DATA_DIR)
        all_files = []
        if data_path.exists() and data_path.is_dir():
            # Get all Word and Excel files
            for ext in [".docx", ".xlsx", ".xls"]:
                all_files.extend(data_path.glob(f"**/*{ext}"))

        # Prepare response
        files = []
        for file_path in all_files:
            filename = file_path.name

            # Check if this file is indexed - using just the filename
            is_indexed = any(source.endswith(filename) for source in indexed_sources)

            # Get file stats
            stats = file_path.stat()

            # Determine file type
            file_type = None
            if filename.lower().endswith(".docx"):
                file_type = "Word Document"
            elif filename.lower().endswith((".xlsx", ".xls")):
                file_type = "Excel Spreadsheet"
            else:
                file_type = "Unknown"

            files.append({
                "filename": filename,
                "path": str(file_path.relative_to(PROJECT_ROOT)),  # Show path relative to project root
                "size_bytes": stats.st_size,
                "last_modified": stats.st_mtime,
                "file_type": file_type,
                "is_indexed": is_indexed
            })

        return files

    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")


@app.delete("/files/{filename}", response_model=Dict[str, Any])
async def delete_file(filename: str):
    """Delete a file and remove it from the index"""
    try:
        # Check if file exists in data directory
        file_path = Path(DATA_DIR) / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {filename}")

        logger.info(f"Deleting file: {filename}")

        # Remove from vector store first (if indexed)
        chunks_removed = vector_store.remove_document(filename)

        # Save updated index if any chunks were removed
        if chunks_removed > 0:
            vector_store.save_index(INDEX_PATH)
            logger.info(f"Removed {chunks_removed} chunks from index")

        # Delete the file
        file_path.unlink()
        logger.info(f"Deleted file: {filename}")

        return {
            "status": "success",
            "filename": filename,
            "chunks_removed": chunks_removed,
            "file_deleted": True
        }

    except Exception as e:
        logger.error(f"Error deleting file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)