# test_quick.py
# import sys
# from pathlib import Path
#
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))

from src.document_processor import ExcelProcessor

processor = ExcelProcessor()
document = processor.process("data/sample_employees.xlsx")

print("✅ Excel processing test:")
print(f"   - Status: {document.metadata.processing_status}")
print(f"   - Sheets: {document.metadata.sheet_names}")
print(f"   - Rows: {document.metadata.total_rows}")
print(f"   - Chunks: {len(document.chunks)}")
print(f"   - Sample chunk: {document.chunks[0].content[:100]}...")

print(f"\nAll chunks:")
for i, chunk in enumerate(document.chunks):
    print(f"Chunk {i+1} (ID: {chunk.chunk_id}):")
    print(f"  Content: {chunk.content}")
    print(f"  Metadata: {chunk.metadata}")
    print()