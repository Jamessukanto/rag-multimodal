"""
PDF page splitting using PyMuPDF
"""

import logging
import hashlib
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from core.exceptions import IngestionError
from storage.document_sql_store import ChunkSource, ChunkLevel

logger = logging.getLogger(__name__)


class PDFSplitter:
    """Splits PDFs into page chunks"""
    
    def __init__(
        self,
        chunk_dir: str,
    ):
        self.chunk_dir = chunk_dir
        Path(chunk_dir).mkdir(parents=True, exist_ok=True)


    def split_pdf_and_store_page_chunks(
        self,
        pdf_id: str,
        pdf_path: str,
        pdf_name: str,
    ) -> List[Dict[str, Any]]:
        """Splits PDFs into page chunks, stores them, and generates chunk metadata"""

        try:
            chunks = []

            with fitz.open(pdf_path) as src_doc:
                # Split entire pdf to its pages
                for i in range(len(src_doc)):
                    page_num = i + 1
                    chunk_id = self._generate_chunk_id(pdf_id, page_num)
                    chunk_name = f"{pdf_name.split('.')[0]}__{page_num}.pdf"
                    chunk_path = f"{self.chunk_dir}/{chunk_name}"

                    with fitz.open() as dest_doc:
                        dest_doc.insert_pdf(
                            src_doc,
                            from_page=i,
                            to_page=i,
                        )
                        dest_doc.save(chunk_path)

                    chunks.append({
                        "chunk_id": chunk_id,
                        "pdf_id": pdf_id,
                        "chunk_name": chunk_name,
                        "chunk_path": chunk_path,
                        "chunk_source": ChunkSource.PDF.value,
                        "chunk_level": ChunkLevel.PAGE.value,
                    })

            logger.info(f"Split PDF {pdf_name} into {len(chunks)} page chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting PDF for pdf_id {pdf_id}: {e}")
            raise IngestionError(f"Failed to split PDF for pdf_id {pdf_id}: {e}")

                    
    def _generate_chunk_id(self, identifier: str, page_number: int) -> str:
        """Generate a unique chunk ID"""
        content = f"{identifier}:{page_number}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

