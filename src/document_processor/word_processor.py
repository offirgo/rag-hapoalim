"""
Word document processor
Converts Word (.docx) files to ProcessedDocument objects with text chunks
"""

import logging
import time
from pathlib import Path
from typing import List, Union, Tuple

from docx import Document
from docx.shared import Inches
from docx.oxml.exceptions import InvalidXmlError
from docx.opc.exceptions import PackageNotFoundError

from src.models.document import (
    DocumentType,
    ProcessingStatus,
    DocumentChunk,
    DocumentMetadata,
    ProcessedDocument,
    ProcessingConfig
)

logger = logging.getLogger(__name__)


class WordProcessingError(Exception):
    """Raised when Word processing fails"""
    pass


class WordProcessor:
    """
    Processes Word (.docx) files into structured document chunks

    Converts document paragraphs into text chunks that can be embedded
    and retrieved by RAG systems.
    """

    def __init__(self, config: ProcessingConfig = None):
        """
        Initialize Word processor with configuration

        Args:
            config: Processing configuration, defaults to ProcessingConfig()
        """
        self.config = config or ProcessingConfig()
        self.config.validate_config()

        logger.info(f"WordProcessor initialized with max_chunk_size={self.config.max_chunk_size}")

    def process(self, file_path: Union[str, Path]) -> ProcessedDocument:
        """
        Process a Word file into a ProcessedDocument

        Args:
            file_path: Path to the Word file

        Returns:
            ProcessedDocument with chunks and metadata

        Raises:
            WordProcessingError: If processing fails
        """
        start_time = time.time()
        file_path = Path(file_path)

        try:
            logger.info(f"Starting Word processing: {file_path}")

            # Validate file
            self._validate_file(file_path)

            # Read Word document
            doc, paragraphs = self._read_word_file(file_path)

            # Create metadata
            metadata = self._create_document_metadata(file_path, doc, paragraphs)
            metadata.processing_status = ProcessingStatus.PROCESSING

            # Process paragraphs into chunks
            chunks = self._create_chunks_from_paragraphs(paragraphs, file_path.name)

            # Update metadata with results
            processing_time = time.time() - start_time
            metadata.total_chunks = len(chunks)
            metadata.processing_time_seconds = processing_time
            metadata.processing_status = ProcessingStatus.COMPLETED

            logger.info(f"Word processing completed: {len(chunks)} chunks in {processing_time:.2f}s")

            return ProcessedDocument(metadata=metadata, chunks=chunks)

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Word processing failed for {file_path}: {str(e)}"
            logger.error(error_msg)

            # Create failed metadata if possible
            try:
                metadata = DocumentMetadata(
                    filename=file_path.name,
                    document_type=DocumentType.WORD,
                    file_size=file_path.stat().st_size if file_path.exists() else 0,
                    processing_status=ProcessingStatus.FAILED,
                    processing_time_seconds=processing_time,
                    error_message=str(e)
                )
                return ProcessedDocument(metadata=metadata, chunks=[])
            except:
                # If even metadata creation fails, raise the original error
                raise WordProcessingError(error_msg) from e

    def _validate_file(self, file_path: Path) -> None:
        """
        Validate that the file exists and is processable

        Args:
            file_path: Path to validate

        Raises:
            WordProcessingError: If validation fails
        """
        if not file_path.exists():
            raise WordProcessingError(f"File does not exist: {file_path}")

        if not file_path.is_file():
            raise WordProcessingError(f"Path is not a file: {file_path}")

        if file_path.stat().st_size == 0:
            raise WordProcessingError(f"File is empty: {file_path}")

        # Check file extension
        allowed_extensions = {'.docx'}
        if file_path.suffix.lower() not in allowed_extensions:
            raise WordProcessingError(
                f"Unsupported file extension '{file_path.suffix}'. "
                f"Supported: {allowed_extensions}"
            )

        # Check if file is readable
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
        except PermissionError:
            raise WordProcessingError(f"Permission denied reading file: {file_path}")
        except Exception as e:
            raise WordProcessingError(f"Cannot read file {file_path}: {e}")

    def _read_word_file(self, file_path: Path) -> Tuple[Document, List[Tuple[str, str, int]]]:
        """
        Read Word file and extract paragraphs with metadata

        Args:
            file_path: Path to Word file

        Returns:
            Tuple of (Document object, List of (paragraph_text, paragraph_type, paragraph_index))

        Raises:
            WordProcessingError: If reading fails
        """
        try:
            # Read Word document
            doc = Document(file_path)

            # Extract paragraphs with metadata
            paragraphs = []
            current_heading = ""

            for i, paragraph in enumerate(doc.paragraphs):
                # Skip completely empty paragraphs if configured to do so
                if self.config.skip_empty_paragraphs and not paragraph.text.strip():
                    continue

                # Determine paragraph type
                para_type = self._get_paragraph_type(paragraph)

                # Track current heading for context
                if para_type.startswith("heading"):
                    current_heading = paragraph.text.strip()

                # Create paragraph info
                para_info = (
                    paragraph.text.strip(),
                    para_type,
                    i,
                    current_heading  # Add heading context
                )
                paragraphs.append(para_info)

            if not paragraphs:
                raise WordProcessingError(f"No readable paragraphs found in document: {file_path}")

            logger.debug(f"Successfully extracted {len(paragraphs)} paragraphs")
            return doc, paragraphs

        except (PackageNotFoundError, InvalidXmlError) as e:
            raise WordProcessingError(f"File appears to be corrupted or not a valid Word document: {e}")
        except Exception as e:
            if isinstance(e, WordProcessingError):
                raise
            raise WordProcessingError(f"Failed to read Word file: {e}")

    def _get_paragraph_type(self, paragraph) -> str:
        """
        Determine the type of paragraph (heading, normal, list, etc.)

        Args:
            paragraph: python-docx Paragraph object

        Returns:
            String describing paragraph type
        """
        style_name = paragraph.style.name.lower()

        if 'heading' in style_name:
            if 'heading 1' in style_name:
                return "heading_1"
            elif 'heading 2' in style_name:
                return "heading_2"
            elif 'heading 3' in style_name:
                return "heading_3"
            else:
                return "heading_other"
        elif 'list' in style_name or 'bullet' in style_name:
            return "list_item"
        elif 'quote' in style_name:
            return "quote"
        elif 'title' in style_name:
            return "title"
        else:
            return "normal"

    def _convert_paragraph_to_text(self, para_text: str, para_type: str, current_heading: str) -> str:
        """
        Convert a paragraph to enhanced natural language text

        DESIGN DECISION: We use a naive natural language approach as a first step.
        Alternative approaches to consider with more data and performance analysis:
        - Structured document representation preserving exact formatting
        - Template-based conversion with document structure awareness
        - Semantic paragraph classification and importance weighting
        - Cross-reference and citation preservation for technical documents

        Example:
            Input:
                para_text: "TechCorp supports flexible working arrangements and remote work options."
                para_type: "normal"
                current_heading: "Work Schedule and Remote Work Policy"

            Output:
                "Under Work Schedule and Remote Work Policy: TechCorp supports flexible working arrangements and remote work options."

        Args:
            para_text: Raw paragraph text
            para_type: Type of paragraph (heading, normal, list_item, etc.)
            current_heading: Current section heading for context

        Returns:
            Enhanced natural language description
        """
        if not para_text.strip():
            return ""

        # Clean the text
        clean_text = " ".join(para_text.split())  # Normalize whitespace

        # Add context based on paragraph type
        if para_type.startswith("heading"):
            # Headings are important structure - include level info
            level = para_type.replace("heading_", "").replace("_", " ")
            return f"Section {level}: {clean_text}"

        elif para_type == "title":
            return f"Document title: {clean_text}"

        elif para_type == "list_item":
            if current_heading:
                return f"Under {current_heading}, item: {clean_text}"
            else:
                return f"List item: {clean_text}"

        elif para_type == "quote":
            if current_heading:
                return f"Quote in {current_heading}: {clean_text}"
            else:
                return f"Quote: {clean_text}"

        else:  # normal paragraphs
            if current_heading:
                return f"Under {current_heading}: {clean_text}"
            else:
                return clean_text

    def _create_chunks_from_paragraphs(
            self,
            paragraphs: List[Tuple[str, str, int, str]],
            source: str
    ) -> List[DocumentChunk]:
        """
        Create document chunks from paragraphs

        NOTE: Paragraphs in Word docs have natural boundaries and context,
        unlike tabular data. We group related paragraphs but preserve
        section boundaries and don't overlap content.

        Args:
            paragraphs: List of (text, type, index, current_heading) tuples
            source: Source filename

        Returns:
            List of DocumentChunk objects
        """
        chunks = []

        # Convert paragraphs to enhanced text
        paragraph_texts = []
        for para_text, para_type, para_idx, current_heading in paragraphs:
            try:
                enhanced_text = self._convert_paragraph_to_text(para_text, para_type, current_heading)
                if enhanced_text.strip():
                    paragraph_texts.append((enhanced_text, para_type, para_idx, current_heading))
            except Exception as e:
                logger.warning(f"Failed to convert paragraph {para_idx}: {e}")
                continue

        if not paragraph_texts:
            logger.warning(f"No valid paragraphs extracted from {source}")
            return chunks

        # Group paragraphs into chunks based on max_chunk_size
        current_chunk_text = ""
        current_chunk_paragraphs = []
        current_chunk_types = set()
        current_chunk_headings = set()
        chunk_index = 0

        for enhanced_text, para_type, para_idx, heading in paragraph_texts:
            # Check if adding this paragraph would exceed chunk size
            potential_chunk = current_chunk_text + "\n" + enhanced_text if current_chunk_text else enhanced_text

            # Special handling for headings - prefer to start new chunks with headings
            should_start_new_chunk = (
                    len(potential_chunk) > self.config.max_chunk_size or
                    (para_type.startswith("heading") and current_chunk_text)  # Start new chunk for headings
            )

            if not should_start_new_chunk:
                # Add to current chunk
                current_chunk_text = potential_chunk
                current_chunk_paragraphs.append(para_idx)
                current_chunk_types.add(para_type)
                if heading:
                    current_chunk_headings.add(heading)
            else:
                # Current chunk is full or we hit a heading, finalize it
                if current_chunk_text:
                    chunk = self._create_chunk(
                        content=current_chunk_text,
                        source=source,
                        chunk_index=chunk_index,
                        paragraph_indices=current_chunk_paragraphs,
                        paragraph_types=list(current_chunk_types),
                        headings=list(current_chunk_headings)
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk with current paragraph
                current_chunk_text = enhanced_text
                current_chunk_paragraphs = [para_idx]
                current_chunk_types = {para_type}
                current_chunk_headings = {heading} if heading else set()

        # Add final chunk if there's remaining content
        if current_chunk_text:
            chunk = self._create_chunk(
                content=current_chunk_text,
                source=source,
                chunk_index=chunk_index,
                paragraph_indices=current_chunk_paragraphs,
                paragraph_types=list(current_chunk_types),
                headings=list(current_chunk_headings)
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(
            self,
            content: str,
            source: str,
            chunk_index: int,
            paragraph_indices: List[int],
            paragraph_types: List[str],
            headings: List[str]
    ) -> DocumentChunk:
        """
        Create a single DocumentChunk with proper metadata

        Args:
            content: Text content of the chunk
            source: Source filename
            chunk_index: Index of this chunk in the document
            paragraph_indices: List of original paragraph indices
            paragraph_types: List of paragraph types in this chunk
            headings: List of section headings relevant to this chunk

        Returns:
            DocumentChunk object
        """
        chunk_id = f"{Path(source).stem}_para_{chunk_index:03d}"

        metadata = {
            "paragraph_indices": paragraph_indices,
            "paragraph_count": len(paragraph_indices),
            "paragraph_types": paragraph_types,
            "section_headings": headings,
            "character_count": len(content),
            "has_headings": any(ptype.startswith("heading") for ptype in paragraph_types)
        }

        return DocumentChunk(
            chunk_id=chunk_id,
            content=content.strip(),
            source=source,
            chunk_index=chunk_index,
            metadata=metadata
        )

    def _create_document_metadata(
            self,
            file_path: Path,
            doc: Document,
            paragraphs: List[Tuple[str, str, int, str]]
    ) -> DocumentMetadata:
        """
        Create document metadata from file and document information

        Args:
            file_path: Path to the Word file
            doc: python-docx Document object
            paragraphs: List of processed paragraphs

        Returns:
            DocumentMetadata object
        """
        # Calculate word count from all paragraph text
        total_word_count = sum(
            len(para_text.split()) for para_text, _, _, _ in paragraphs
            if para_text.strip()
        )

        return DocumentMetadata(
            filename=file_path.name,
            document_type=DocumentType.WORD,
            file_size=file_path.stat().st_size,
            word_count=total_word_count,
            total_rows=len(paragraphs),  # Use total_rows for paragraph count for consistency
            processing_status=ProcessingStatus.PENDING
        )