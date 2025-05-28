import tempfile
import os
import hashlib
import logging
import gzip
import magic # For python-magic
from typing import List, Tuple

# Langchain and Unstructured related imports
from langchain_unstructured import UnstructuredLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel # To be used as a type hint for Langchain Document

# Project-specific imports
from .config import settings
from .models import DocumentItem

# Note: The Langchain Document class is typically imported from langchain_core.documents
# However, the existing code in split.py uses `from pydantic import BaseModel`
# and then type hints Langchain documents as `BaseModel`. We'll follow that pattern
# for consistency for now, but ideally, it should be `from langchain_core.documents import Document as LangchainDocument`
# and used as `LangchainDocument`. For this refactoring, I'll stick to `BaseModel` as the type hint
# where Langchain documents are expected.

logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Custom exception for errors during document loading or processing."""
    pass


def is_gz_file(file_path: str) -> bool:
    """
    Checks if a file is GZip compressed by inspecting its first two bytes.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        bool: True if the file is GZip compressed, False otherwise.
    """
    with open(file_path, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'


def get_mime_type(file_path: str) -> str:
    """
    Determines the MIME type of a file using the python-magic library.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The detected MIME type of the file.
    """
    mime_type = magic.from_file(file_path, mime=True)
    return mime_type


def load_by_unstructured(file_path: str) -> List[BaseModel]:
    """
    Loads a document using UnstructuredLoader with specific post-processors
    and a basic chunking strategy.

    This function configures the UnstructuredLoader to clean extra whitespace
    and uses a "basic" chunking strategy with a very large character limit
    to effectively load the document as a whole before further splitting.

    Args:
        file_path (str): The path to the file to load.

    Returns:
        List[BaseModel]: A list of Langchain Document objects.
    
    Raises:
        DocumentProcessingError: If UnstructuredLoader fails to process the document.
    """
    try:
        loader = UnstructuredLoader(
                    file_path=file_path,
                    post_processors=[clean_extra_whitespace],
                    chunking_strategy="basic", # This aims to load the whole doc content
                    max_characters=10000000, # Very large to avoid chunking by Unstructured itself
                    include_orig_elements=False,
                )
        return loader.load()
    except Exception as e:
        logger.error(f"UnstructuredLoader failed for path {file_path}: {e}", exc_info=True)
        raise DocumentProcessingError(f"Failed to process document with UnstructuredLoader: {str(e)}")


def load(uploaded_file_path: str, settings_obj) -> Tuple[List[BaseModel], str]:
    """
    Loads a document from a temporary file, handling GZip decompression if necessary,
    and processes it using UnstructuredLoader.

    Args:
        uploaded_file_path (str): The path to the (potentially gzipped) uploaded file.
        settings_obj: The application settings object, used to access delete_temp_file.

    Returns:
        Tuple[List[BaseModel], str]: A tuple where the first element is a list of
            Langchain Document objects, and the second element is the detected
            MIME type of the processed file content.
            
    Raises:
        DocumentProcessingError: If GZip decompression or document processing fails.
    """
    if is_gz_file(uploaded_file_path):
        logger.info(f'File {uploaded_file_path} is gzip compressed. Decompressing...')
        decompressed_file_path = None
        try:
            # Create a new temporary file for the decompressed content
            with tempfile.NamedTemporaryFile(
                    mode='wb', delete=False, suffix=".gz_decompressed") as decompressed_file_obj:
                decompressed_file_path = decompressed_file_obj.name
                with gzip.open(uploaded_file_path, 'rb') as gzipped_file:
                    while True:
                        chunk = gzipped_file.read(1024 * 1024) # Read in 1MB chunks
                        if not chunk:
                            break
                        decompressed_file_obj.write(chunk)
            
            # Now load from the decompressed file path
            docs = load_by_unstructured(decompressed_file_path)
            mime_type = get_mime_type(decompressed_file_path)
            return docs, mime_type
        except gzip.BadGzipFile as e:
            logger.error(f"Gzip decompression failed for {uploaded_file_path}: {e}", exc_info=True)
            raise DocumentProcessingError(f"Failed to decompress GZip file: {str(e)}")
        except Exception as e: # Catch other errors during decompressed load
            logger.error(f"Error processing decompressed file from {uploaded_file_path} (temp path {decompressed_file_path}): {e}", exc_info=True)
            raise DocumentProcessingError(f"Failed to process decompressed file: {str(e)}")
        finally:
            if decompressed_file_path and settings_obj.delete_temp_file and os.path.exists(decompressed_file_path):
                os.remove(decompressed_file_path)
    else:
        mime_type = get_mime_type(uploaded_file_path)
        # Pass the original path directly to load_by_unstructured
        docs = load_by_unstructured(uploaded_file_path)
        return docs, mime_type


def get_doc_id(doc: BaseModel) -> str:
    """
    Generates a unique 12-character ID for a document based on an MD5 hash
    of its 'source' metadata.

    Args:
        doc (BaseModel): A Langchain Document object which must have a 'metadata'
            attribute containing a 'source' key.

    Returns:
        str: A 12-character unique ID for the document.
    """
    current_md5 = hashlib.md5()
    # Ensure metadata and source exist and are strings
    source_material = ""
    if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) and 'source' in doc.metadata:
        source_material = str(doc.metadata['source'])
    
    current_md5.update(source_material.encode('utf-8'))
    uid = current_md5.hexdigest()[:12]
    return uid


def split(doc: BaseModel, q_chunk_size: int, q_chunk_overlap: int) -> List[DocumentItem]:
    """
    Splits a Langchain Document into smaller DocumentItem chunks using
    RecursiveCharacterTextSplitter.

    Args:
        doc (BaseModel): The Langchain Document object to be split. Expected to have
            `page_content` (str) and `metadata` (dict) attributes.
        q_chunk_size (int): The target maximum size for each chunk in characters.
        q_chunk_overlap (int): The number of characters to overlap between
            consecutive chunks.

    Returns:
        List[DocumentItem]: A list of DocumentItem Pydantic models.
    """
    items = []
    # Ensure page_content exists and is a string
    page_content = getattr(doc, 'page_content', '')
    if not isinstance(page_content, str):
        page_content = ""

    if len(page_content) > q_chunk_size:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=q_chunk_size,
            chunk_overlap=q_chunk_overlap,
            length_function=len,
            add_start_index=True, # Useful for some applications
        )
        texts = text_splitter.split_text(page_content)
        logger.debug(f"Number of chunks created: {len(texts)}")
        if len(texts) > 1:
            logger.debug(f"Sample chunk [1] (first 100 chars): {texts[1][:100]}...")
        
        doc_metadata = getattr(doc, 'metadata', {})
        if not isinstance(doc_metadata, dict): # Ensure metadata is a dict
            doc_metadata = {}
            
        # The original 'id' generation based on document source was problematic for chunks.
        # Each chunk should ideally get its own ID or retain source ID + chunk sequence.
        # For now, we'll use the source ID for all chunks from that source, plus add chunk index.
        # A more robust solution might involve hashing chunk content for a unique chunk ID.
        
        source_doc_id = get_doc_id(doc) # Get ID from document source
        
        for i, text_chunk in enumerate(texts):
            chunk_metadata = doc_metadata.copy() # Start with original doc metadata
            chunk_metadata['id'] = f"{source_doc_id}_chunk_{i}" # Append chunk index to source ID
            chunk_metadata['chunk_index'] = i
            # Potentially add start_index from splitter if `add_start_index=True` was used
            # and the splitter provides this information in a structured way.
            # text_splitter.split_documents might be better if metadata per chunk is needed.

            items.append(
                DocumentItem(
                    content=text_chunk,
                    metadata=chunk_metadata,
                )
            )
    elif page_content: # If content exists but isn't larger than chunk_size
        doc_metadata = getattr(doc, 'metadata', {})
        if not isinstance(doc_metadata, dict):
            doc_metadata = {}
        doc_metadata['id'] = f"{get_doc_id(doc)}_chunk_0"
        doc_metadata['chunk_index'] = 0
        items.append(
            DocumentItem(
                content=page_content,
                metadata=doc_metadata,
            )
        )
    else: # Content is not larger than chunk_size
        if page_content: # If there's content, but it's not split
             logger.debug(f"Content length {len(page_content)} <= chunk_size {q_chunk_size}. Full content not split, creating one item.")
             # This part was already correctly creating one item if page_content was not empty and not split.
        else: # page_content is empty
            logger.debug("Empty page_content, no items created from splitting.")

    # If page_content is empty, items list will be empty.
    return items
