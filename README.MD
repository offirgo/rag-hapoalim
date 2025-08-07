# RAG System

A Retrieval-Augmented Generation (RAG) system that processes Word and Excel documents to answer questions based on document content.

## Overview

This system allows you to:
- Upload Word (.docx) and Excel (.xlsx) files
- Ask questions about the content in these documents
- Get AI-generated answers based specifically on your documents

## Installation

### Prerequisites

- Python 3.8+
- pip
- git

### Step 1: Clone the Repository

```bash
    git clone https://github.com/yourusername/rag-hapoalim.git
    cd rag-hapoalim
```

### Step 2: Install Python Dependencies

```bash
  pip install -r requirements.txt
```

### Step 3: Install Ollama (Optional for real LLM use)
If you want to use a real LLM instead of the test mode:

```bash
  # On macOS
  brew install ollama

    # On Linux
    curl -fsSL https://ollama.com/install.sh | sh
    
    # On Windows
    # Download from https://ollama.com/download
```

### Step 4: Download a Model (Optional)
If you want to use a real LLM instead of the test mode:

```bash
  # Start Ollama server in a terminal (keep this running)
    ollama serve
    
    # In a new terminal, download the TinyLlama model
    ollama pull tinyllama
```

## Running the System

###  Download a Model (Optional)
If you want to use a real LLM instead of the test mode:

```bash
    # Navigate to the project directory
    cd rag-hapoalim
    
    # Start the API server
    python src/api/api.py
```

The API will be available at http://localhost:8000, and the interactive documentation at http://localhost:8000/docs.

### Using the Interactive API Documentation

Open http://localhost:8000/docs in your browser
Try the following endpoints:

GET /info - system info
GET /files - List available files
POST /upload - Upload a document (only .docx, .xlsx, and .xls are supported))
POST /answer - Ask a question about your documents
POST /files/{filename} - Delete a file from both data folder and index

## question and ansewr examples
question: What is the company policy on remote work?

answer: Based on the information provided in Passages 1, 2, and 3, the company's IT and security policies require that employees use company-approved software and strictly follow security protocols. Devices must be updated with antivirus software and have their operating system patches applied regularly. Personal devices are only used for work under approval from IT and security teams, and two-factor authentication is required for all company systems. Employees must also follow this policy when working remotely.

question: Who has the highest salary in the Engineering department?

answer: Based on the provided information, the highest salary in the Engineering department is not specified as it does not directly relate to Department_Summary or Employee_Summary. The given information does indicate that there are three employees with high salaries, namely EMPI001 John Smith, EMPI002 Emily Chen, and EMPI003 Michael Rodriguez. However, the specific department and manager of these employees is not specified, making it impossible to determine which department they work in or who the manager is

## Architecture & Technical Design

### System Architecture

The system follows a modular architecture with these key components:

1. **Document Processors**
   - **Word Processor**: Extracts text from .docx files while preserving document structure
   - **Excel Processor**: Converts tabular data to text with row/column context
   - Both implement chunking strategies to create semantically meaningful text segments

2. **Embedding Generation**
   - Uses SentenceTransformers for local, offline embedding generation
   - Converts text chunks to 384-dimensional vector representations
   - Normalizes vectors for cosine similarity operations

3. **Vector Storage (FAISS)**
   - Stores document embeddings with metadata
   - Enables efficient similarity search
   - Persists index to disk for reuse
   - Supports document addition and removal

4. **LLM Integration**
   - Supports multiple LLM backends:
     - Local models via Ollama (TinyLlama)
     - Test mode (EchoLLM) for debugging without dependencies
   - Abstracts LLM details behind a common interface

5. **RAG Service**
   - Coordinates retrieval and generation processes
   - Constructs prompts with retrieved context
   - Routes queries to appropriate LLM
   - Tracks performance metrics

6. **API Layer**
   - RESTful endpoints for all operations
   - File upload and management
   - Interactive documentation
   - Request validation and error handling

### Data Flow

1. **Ingestion Pipeline**:
   - Document uploaded → Document processor → Chunking → Embedding generation → Vector storage
   
2. **Query Pipeline**:
   - Question → Embedding → Vector retrieval → Context assembly → LLM prompt construction → Response generation

### Technical Choices

- **FAISS**: Facebook AI Similarity Search provides high-performance vector operations. We chose the flat index for simplicity and accuracy, which is suitable for collections with thousands of chunks.

- **SentenceTransformers**: Provides high-quality embeddings without external API dependencies. The all-MiniLM-L6-v2 model offers an excellent balance of quality and performance with 384-dimensional embeddings.

- **Ollama**: Simplifies running LLMs locally with minimal setup. TinyLlama provides reasonable performance on consumer hardware while maintaining good response quality.

- **FastAPI**: Modern API framework with automatic documentation, type validation, and asynchronous support. Significantly reduces boilerplate compared to Flask.

- **Pydantic**: Handles data validation, serialization, and documentation in a type-safe manner, ensuring robust API interfaces.

### System Requirements

- **Storage**: ~500MB for code, dependencies, and embeddings model
- **Memory**: 
  - Minimum: 4GB RAM
  - Recommended: 8GB+ RAM (especially when using Ollama)
- **Processing**:
  - CPU-only operation is supported
  - GPU acceleration used automatically when available

### Performance Considerations

- **Indexing**: Processing and embedding generation is the most computationally intensive part of the pipeline. Performance scales with document size.

- **Retrieval**: Vector search is efficient even with thousands of chunks, typically completing in <100ms.

- **Generation**: Response time depends on the LLM choice:
  - EchoLLM: Near-instant (<10ms)
  - TinyLlama: 1-5 seconds depending on hardware

## Future Improvements

The current implementation provides a solid foundation for a RAG system, but there are several areas for future enhancement:

### Embeddings

- **Cloud Embeddings**: Add support for OpenAI, Anthropic, and Cohere embedding models
- **Performance Optimization**:
  - Implement batch processing for better performance with large documents
  - Add model caching and lazy loading for faster startup
  - Consider text preprocessing (truncation, cleaning)
- **Versatility**:
  - Support different models based on use case (multilingual, domain-specific)
  - Add embedding normalization options and make them configurable

### Vector Storage

- **Scalability**:
  - Support different FAISS index types (IVF, HNSW) for large datasets
  - Implement batch operations for better performance with many chunks
- **Advanced Features**:
  - Add index optimization and compression options
  - Support for filtering during search (not just post-filtering)
  - Add support for updating individual chunks without full rebuild

### Document Processing

- **Memory Efficiency**: Change Excel processing to stream for better memory usage
- **Support More Formats**: Add processors for PDF, CSV, and plain text files

### LLM Integration

- **More Models**: Add support for other LLM providers (Anthropic, Cohere, etc.)
- **Advanced Prompting**: Implement more sophisticated prompt engineering techniques
- **Parameter Tuning**: Add UI for adjusting generation parameters

### API and UI

- **User Interface**: Develop a web-based UI for non-technical users
- **Authentication**: Add user authentication and document access control
- **Rate Limiting**: Implement rate limiting for production use