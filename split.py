from fastapi import FastAPI, APIRouter, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from validation_uploadfile import ValidateUploadFileMiddleware

# Langchain imports are no longer directly used here if logic is in core_processing
# from langchain_unstructured import UnstructuredLoader
# from langchain_community.document_loaders import PyMuPDFLoader # Example if it was used
# from unstructured.cleaners.core import clean_extra_whitespace
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# Pydantic BaseModel might still be needed for type hints if Langchain docs are hinted as such
# from pydantic import BaseModel # No longer needed if all models are in models.py and not used for type hints here
from typing import List, Optional # Retain for Optional in load_split if needed, List for SplitConfig return type
import tempfile # Still needed for NamedTemporaryFile in load_split
import os # Still needed for uvicorn.run and Mangum handler logic
import logging
from fastapi import HTTPException # For raising HTTP exceptions
import nltk # For nltk.data.path

logger = logging.getLogger(__name__)

# Removed: hashlib, gzip, magic (moved to core_processing.py)

from .config import settings
from .models import DocumentItem, Document, SplitConfig # Import Pydantic models
from .core_processing import load, split, DocumentProcessingError # Import core functions and custom exception

# Set the nltk.data.path with environment variable
if settings.nltk_data and settings.nltk_data not in nltk.data.path:
    nltk.data.path.append(settings.nltk_data)

router = APIRouter()


def create_app():
    """
    Initializes and returns the FastAPI application.

    This function sets up the FastAPI application instance, configures CORS (Cross-Origin
    Resource Sharing) middleware, adds the ValidateUploadFileMiddleware for input validation,
    registers a GZipMiddleware for compressing responses, and includes the API routers.

    Returns:
        FastAPI: The configured FastAPI application instance.
    """
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        ValidateUploadFileMiddleware,
        app_paths=["/split"], # Changed to list as per task
        # 1000000 is 1MB for storage, 1048576 is 1MB for memory
        # REF:
        max_size=settings.max_file_size_in_mb * 1000000,
        file_types=settings.supported_file_types
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000) # Default minimum_size for Gzip is 1KB

    app.include_router(router)

    return app


# Functions is_gz_file, get_mime_type, load, load_by_unstructured, get_doc_id, split
# have been moved to core_processing.py

# Pydantic models DocumentItem, Document, SplitConfig have been moved to models.py


@router.get(
    "/split/config",
    response_model=SplitConfig,
    description="Get the current configurations for the split endpoint"
)
async def get_config():
    """
    Endpoint handler to return the current service configuration.

    Retrieves various configuration parameters (like file size limits,
    chunking parameters, supported types) from environment variables
    and returns them in a structured Pydantic model.

    Returns:
        SplitConfig: A Pydantic model instance containing the current
            service configuration settings.
    """
    return SplitConfig(
        delete_temp_file=settings.delete_temp_file,
        nltk_data=settings.nltk_data,
        max_file_size_in_mb=settings.max_file_size_in_mb,
        supported_file_types=settings.supported_file_types,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap
    )


@router.post(
    "/split",
    response_model=Document,
)
async def load_split(
    file: UploadFile = File(...),
    q_chunk_size: int = Query(
        settings.chunk_size, description='Maximum size of chunks in characters to return'),
    q_chunk_overlap: int = Query(
        settings.chunk_overlap, description='Overlap in characters between chunks')
):
    """
    Endpoint handler to accept an uploaded file, process it through loading
    and splitting, and return the structured document chunks.

    The file is first saved to a temporary location. It's then loaded using
    `load()` (which handles GZip and uses `UnstructuredFileLoader`), and the
    resulting Langchain Document(s) are split into chunks using `split()`.
    The chunks are then formatted into the `Document` Pydantic response model.

    Args:
        file (UploadFile): The uploaded file to be processed. FastAPI handles
            receiving this from the `multipart/form-data` request.
        q_chunk_size (int, optional): The target maximum size for each chunk
            in characters. Defaults to the `CHUNK_SIZE` environment variable.
        q_chunk_overlap (int, optional): The number of characters to overlap
            between consecutive chunks. Defaults to the `CHUNK_OVERLAP`
            environment variable.

    Returns:
        Document: A Pydantic model instance containing the processed document's
            MIME type and a list of its textual chunks with metadata.

    Raises:
        HTTPException: Can raise a 500 internal server error if any unexpected
            issue occurs during file processing. Can also raise 422 if specific
            DocumentProcessingError occurs. More specific HTTP errors might
            be raised by the `ValidateUploadFileMiddleware` before this handler
            is reached.
    """
    file_chunk_size = 1024 * 1024  # 1 MB
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='wb',
                buffering=file_chunk_size,
                delete=False, # Manage deletion manually in finally block
                suffix=f"_{file.filename}" # Add suffix for easier identification
                ) as temp:
            temp_file_path = temp.name
            while True:
                chunk = await file.read(file_chunk_size)
                if not chunk:
                    break
                temp.write(chunk)
            temp.flush() # Ensure all data is written before load reads it

            # Pass the file path and settings object to the load function
            docs, mime_type = load(temp_file_path, settings)
        
        # Ensure docs is a list, even if it's empty (as per core_processing.load_by_unstructured)
        if not isinstance(docs, list):
            logging.error(f"Loaded documents are not a list: {type(docs)}. Filename: {file.filename}")
            raise DocumentProcessingError("Internal error: Document loading returned unexpected type.")

        docs_len = len(docs)
        logger.info(f"File: {file.filename}, MIME type: {mime_type}, Pages/Docs loaded: {docs_len}")

        all_items = []
        combined_content = [] # To store page_content from all Langchain docs

        if docs_len > 0:
            for doc_idx, doc_content in enumerate(docs): # Use a more descriptive name than 'doc'
                # Ensure doc_content is a Pydantic BaseModel or has page_content
                if not hasattr(doc_content, 'page_content') or not hasattr(doc_content, 'metadata'):
                    logging.warning(f"Skipping invalid document structure at index {doc_idx} for file {file.filename}. Type: {type(doc_content)}")
                    continue

                # Accumulate page content for the top-level 'content' field if needed
                # (though this behavior might be better suited inside core_processing.load if always desired)
                if hasattr(doc_content, 'page_content') and isinstance(doc_content.page_content, str):
                    combined_content.append(doc_content.page_content)
                
                # Call core_processing.split for each document loaded
                # The split function now expects a Langchain Document-like object
                # (which has page_content and metadata)
                current_items = split(doc_content, q_chunk_size, q_chunk_overlap)
                all_items.extend(current_items)
        
        # Determine top-level content: join all page_contents if multiple, else use single or None
        final_content = "".join(combined_content) if combined_content else None
        if mime_type == "text/plain" and docs_len == 1 and hasattr(docs[0], 'page_content'):
             # For plain text with a single document, the original logic might have set content to None.
             # Let's clarify this: if it's plain text, usually the content is the text itself.
             # The original split.py had: content=doc.page_content if mime_type != "text/plain" else None
             # This seems counterintuitive for plain text. Let's assume content should be the text.
             pass # final_content is already set

        document_response = Document(
            content=final_content, # Or adjust based on desired logic for original content
            mime_type=mime_type,
            items=all_items,
        )
        return document_response

    except DocumentProcessingError as e:
        logging.error(f"Document processing error for file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException: # Re-raise HTTPException if it's already one (e.g. from middleware)
        raise
    except Exception as e:
        logging.error(f"Unexpected error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if temp_file_path and settings.delete_temp_file and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Successfully deleted temporary file: {temp_file_path}")
            except Exception as e_remove:
                logger.error(f"Failed to delete temporary file {temp_file_path}: {e_remove}", exc_info=True)


if __name__ == "__main__":
    if settings.runtime == "aws-lambda":
        from mangum import Mangum
        handler = Mangum(create_app())
    else:
        import uvicorn
        uvicorn.run(
            create_app(), host=settings.host, port=settings.port
        )
elif settings.runtime == "aws-lambda": # Added for consistency if RUNTIME is set outside __main__
    from mangum import Mangum
    handler = Mangum(create_app())
