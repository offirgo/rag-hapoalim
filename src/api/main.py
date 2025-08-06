Suggested Tech Stack (we can adjust based on your preferences):

Framework: FastAPI (simple, fast, great docs)
Document Processing: python-docx for Word, pandas + openpyxl for Excel
Embeddings: OpenAI embeddings or sentence-transformers (local)
Vector DB: FAISS (simple to start) or Qdrant (more features)
LLM: OpenAI GPT or Ollama (local)
Chunking: LangChain or custom implementation

Stage 1: Basic document loading and chunking
Stage 2: Embedding generation and vector storage
Stage 3: Simple retrieval system
Stage 4: LLM integration for answer generation
Stage 5: API wrapper with FastAPI
Stage 6: Polish and additional features

- sentence-transformers (local embeddings)
- FAISS (local vector storage)
- pandas + openpyxl (Excel processing)
- python-docx (Word processing)
- FastAPI (API layer)
- Ollama (optional local LLM)