#!/usr/bin/env python3
"""
PDF and Document Ingestion for RAG
Extracts text from PDFs and documents, chunks them, and creates searchable index
"""

import pathlib
import json
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings
            for i in range(end, max(start + chunk_size - 200, start), -1):
                if text[i] in '.!?':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else end
    
    return chunks

def extract_pdf_text(pdf_path: pathlib.Path) -> str:
    """Extract text from PDF using pypdf"""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        text_parts = []
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        return "\\n".join(text_parts)
    except ImportError:
        print("⚠️ pypdf not installed. Install with: pip install pypdf")
        return ""
    except Exception as e:
        print(f"❌ Error extracting PDF {pdf_path.name}: {e}")
        return ""

def extract_text_from_file(file_path: pathlib.Path) -> str:
    """Extract text from various file types"""
    suffix = file_path.suffix.lower()
    
    if suffix == '.pdf':
        return extract_pdf_text(file_path)
    elif suffix in ['.txt', '.md', '.py', '.js', '.ts', '.json', '.yaml', '.yml']:
        try:
            return file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
            return ""
    else:
        print(f"⚠️ Unsupported file type: {suffix}")
        return ""

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\\w\\s.,!?;:\\-\\(\\)\\[\\]{}"\']', ' ', text)
    # Remove excessive spaces
    text = re.sub(r' +', ' ', text)
    return text.strip()

def create_document_record(file_path: pathlib.Path, chunk_text: str, chunk_index: int) -> Dict[str, Any]:
    """Create a document record for the index"""
    # Create unique ID
    content_hash = hashlib.sha256(f"{file_path.name}:{chunk_index}:{chunk_text[:100]}".encode()).hexdigest()[:16]
    
    return {
        "id": content_hash,
        "source_file": file_path.name,
        "source_path": str(file_path),
        "chunk_index": chunk_index,
        "text": chunk_text,
        "text_length": len(chunk_text),
        "created_at": datetime.now().isoformat(),
        "file_size": file_path.stat().st_size if file_path.exists() else 0,
        "file_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
    }

def process_directory(input_dir: pathlib.Path, output_file: pathlib.Path, 
                     file_patterns: List[str] = ["*.pdf", "*.txt", "*.md", "*.py"],
                     chunk_size: int = 1200) -> Dict[str, Any]:
    """Process all files in directory and create RAG index"""
    
    print(f"🔍 Processing directory: {input_dir}")
    print(f"📁 Output file: {output_file}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    all_files = []
    for pattern in file_patterns:
        all_files.extend(input_dir.glob(f"**/{pattern}"))
    
    print(f"📄 Found {len(all_files)} files to process")
    
    # Process files
    processed_count = 0
    total_chunks = 0
    
    with output_file.open('w', encoding='utf-8') as f:
        for file_path in all_files:
            print(f"📖 Processing: {file_path.name}")
            
            # Extract text
            raw_text = extract_text_from_file(file_path)
            if not raw_text:
                print(f"⚠️ No text extracted from {file_path.name}")
                continue
            
            # Clean text
            clean_text_content = clean_text(raw_text)
            if len(clean_text_content) < 50:  # Skip very short texts
                print(f"⚠️ Text too short in {file_path.name}")
                continue
            
            # Create chunks
            chunks = chunk_text(clean_text_content, chunk_size)
            print(f"  📝 Created {len(chunks)} chunks")
            
            # Create records
            for i, chunk in enumerate(chunks):
                record = create_document_record(file_path, chunk, i)
                f.write(json.dumps(record, ensure_ascii=False) + "\\n")
                total_chunks += 1
            
            processed_count += 1
    
    # Create summary
    summary = {
        "processed_files": processed_count,
        "total_chunks": total_chunks,
        "input_directory": str(input_dir),
        "output_file": str(output_file),
        "chunk_size": chunk_size,
        "created_at": datetime.now().isoformat(),
        "file_patterns": file_patterns
    }
    
    # Save summary
    summary_file = output_file.with_suffix('.summary.json')
    with summary_file.open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Processing complete!")
    print(f"📊 Files processed: {processed_count}")
    print(f"📊 Total chunks: {total_chunks}")
    print(f"📊 Summary saved to: {summary_file}")
    
    return summary

def search_index(index_file: pathlib.Path, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Simple text search in the index (for testing)"""
    if not index_file.exists():
        print(f"❌ Index file not found: {index_file}")
        return []
    
    query_lower = query.lower()
    results = []
    
    with index_file.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                text_lower = record["text"].lower()
                
                # Simple scoring based on term frequency
                score = text_lower.count(query_lower)
                if score > 0:
                    record["relevance_score"] = score
                    results.append(record)
            except Exception:
                continue
    
    # Sort by relevance and return top k
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_k]

def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents for RAG")
    parser.add_argument("input_dir", help="Input directory containing documents")
    parser.add_argument("--output", "-o", default="data/rag_index.jsonl", help="Output index file")
    parser.add_argument("--chunk-size", type=int, default=1200, help="Chunk size in characters")
    parser.add_argument("--search", "-s", help="Search query to test the index")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    args = parser.parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    output_file = pathlib.Path(args.output)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return 1
    
    if args.search:
        # Search mode
        results = search_index(output_file, args.search, args.top_k)
        print(f"🔍 Search results for: '{args.search}'")
        print(f"📊 Found {len(results)} results\\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['source_file']} (chunk {result['chunk_index']})")
            print(f"   Score: {result['relevance_score']}")
            print(f"   Text: {result['text'][:200]}...")
            print()
    else:
        # Process mode
        summary = process_directory(input_dir, output_file, chunk_size=args.chunk_size)
        print(f"\\n🎉 Index created successfully!")
        print(f"Test search with: python {__file__} {input_dir} --search 'your query'")

if __name__ == "__main__":
    main()