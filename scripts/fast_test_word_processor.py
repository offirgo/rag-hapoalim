"""
Manual test of WordProcessor with sample data
Run this to see if our Word processor works before writing formal tests
"""

import logging
from pathlib import Path

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Import our classes
from src.models.document import ProcessingConfig
from src.document_processor import WordProcessor


def test_word_processor():
    """Test Word processor with sample company handbook"""

    print("🧪 Testing WordProcessor with sample data")
    print("=" * 50)

    # Create processor with default config
    config = ProcessingConfig(max_chunk_size=800)  # Smaller chunks for testing
    processor = WordProcessor(config)

    # Test with sample data
    word_file = Path("../data/company_handbook.docx")

    if not word_file.exists():
        print(f"❌ Sample file not found: {word_file}")
        print("Run 'python create_sample_data.py' first!")
        return

    try:
        # Process the file
        print(f"📄 Processing file: {word_file}")
        document = processor.process(word_file)

        # Show results
        print(f"\n✅ Processing completed!")
        print(f"📊 Metadata:")
        print(f"   - Filename: {document.metadata.filename}")
        print(f"   - File size: {document.metadata.file_size} bytes")
        print(f"   - Document type: {document.metadata.document_type}")
        print(f"   - Processing status: {document.metadata.processing_status}")
        print(f"   - Word count: {document.metadata.word_count}")
        print(f"   - Paragraph count: {document.metadata.total_rows}")  # total_rows = paragraph count
        print(f"   - Processing time: {document.metadata.processing_time_seconds:.2f}s")
        print(f"   - Total chunks: {document.metadata.total_chunks}")

        print(f"\n📝 Chunks ({len(document.chunks)}):")
        for i, chunk in enumerate(document.chunks[:3]):  # Show first 3 chunks
            print(f"   Chunk {i + 1} (ID: {chunk.chunk_id}):")
            print(f"   - Content: {chunk.content[:150]}...")
            print(f"   - Metadata: {chunk.metadata}")
            print()

        if len(document.chunks) > 3:
            print(f"   ... and {len(document.chunks) - 3} more chunks")

        # Test properties
        print(f"\n🔍 Document properties:")
        print(f"   - Chunk count: {document.chunk_count}")
        print(f"   - Is processed: {document.is_processed}")
        print(f"   - Chunks with embeddings: {len(document.get_chunks_with_embeddings())}")

        # Show paragraph types found
        all_types = set()
        all_headings = set()
        for chunk in document.chunks:
            all_types.update(chunk.metadata.get("paragraph_types", []))
            all_headings.update(chunk.metadata.get("section_headings", []))

        print(f"\n📋 Document structure:")
        print(f"   - Paragraph types found: {sorted(list(all_types))}")
        print(f"   - Section headings: {len(all_headings)}")
        for heading in sorted(list(all_headings))[:5]:  # Show first 5 headings
            print(f"     • {heading}")

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


def test_word_error_handling():
    """Test error handling with invalid files"""

    print("\n🧪 Testing Word Error Handling")
    print("=" * 50)

    processor = WordProcessor()

    # Test non-existent file
    print("Testing non-existent file...")
    document = processor.process("nonexistent.docx")
    print(f"   Status: {document.metadata.processing_status}")
    print(f"   Error: {document.metadata.error_message}")

    # Test invalid file type
    print("\nTesting invalid file type...")
    try:
        # Create a dummy text file
        Path("dummy.docx").write_text("This is not a real Word file")
        document = processor.process("dummy.docx")
        print(f"   Status: {document.metadata.processing_status}")
        print(f"   Error: {document.metadata.error_message}")
        Path("dummy.docx").unlink()  # Clean up
    except Exception as e:
        print(f"   Exception: {e}")
        # Clean up if error occurred
        if Path("dummy.docx").exists():
            Path("dummy.docx").unlink()


def test_word_paragraph_conversion():
    """Test paragraph-to-text conversion logic"""

    print("\n🧪 Testing Paragraph Conversion")
    print("=" * 50)

    processor = WordProcessor()

    # Test different paragraph types
    test_cases = [
        ("Company Overview", "heading_1", ""),
        ("TechCorp is a leading technology company.", "normal", "Company Overview"),
        ("Health insurance with 90% coverage", "list_item", "Employee Benefits"),
        ("Work-life balance is important.", "quote", ""),
    ]

    for para_text, para_type, heading in test_cases:
        converted = processor._convert_paragraph_to_text(para_text, para_type, heading)
        print(f"   Input: '{para_text}' ({para_type}, heading: '{heading}')")
        print(f"   Output: '{converted}'")
        print()


if __name__ == "__main__":
    test_word_processor()
    test_word_error_handling()
    test_word_paragraph_conversion()
    print("\n🎉 Manual Word testing completed!")