from pydantic import BaseModel
from typing import List, Mapping, Any, Optional

class DocumentItem(BaseModel):
    """
    Represents a single chunk of text extracted from a document, along with its metadata.

    Attributes:
        content (str): The textual content of the chunk.
        metadata (Mapping[str, Any]): A dictionary containing metadata associated
            with the chunk (e.g., source document, page number, chunk ID).
    """
    content: str
    metadata: Mapping[str, Any]


class Document(BaseModel):
    """
    Represents the processed document, including its original content (optional),
    MIME type, and a list of DocumentItem chunks.

    Attributes:
        content (Optional[str]): The original content of the document. This might be
            None, especially for non-plain-text files or if not explicitly populated.
        mime_type (str): The detected MIME type of the uploaded file.
        items (List[DocumentItem]): A list of DocumentItem objects, each representing
            a chunk of the processed document.
    """
    content: Optional[str]
    mime_type: str
    items: List[DocumentItem]

class SplitConfig(BaseModel):
    """
    Represents the service's configuration settings as returned by the /split/config endpoint.

    Attributes:
        delete_temp_file (bool): Indicates if temporary files are deleted after processing.
        nltk_data (str | None): The path to the NLTK data directory.
        max_file_size_in_mb (float): Maximum allowed file size for uploads in MB.
        supported_file_types (List[str]): A list of supported MIME types for document uploads.
        chunk_size (int): The default target size for text chunks in characters.
        chunk_overlap (int): The default number of characters to overlap between chunks.
    """
    delete_temp_file: bool
    nltk_data: Optional[str] = None # Made Optional to align with Pydantic best practices if it can be None
    max_file_size_in_mb: float
    supported_file_types: List[str]
    chunk_size: int
    chunk_overlap: int
