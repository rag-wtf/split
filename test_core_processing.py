import pytest
from unittest.mock import patch, MagicMock, mock_open, call
import tempfile
import os
import gzip
import hashlib

# Functions to be tested
from core_processing import (
    is_gz_file,
    get_mime_type,
    load_by_unstructured,
    load,
    get_doc_id,
    split
)

# Imports for constructing test inputs or expected outputs
from config import settings # To potentially override or use settings in tests
from models import DocumentItem

# Import for Langchain Document type
from langchain_core.documents import Document as LangchainDocument


# --- Tests for is_gz_file ---
@patch("builtins.open", new_callable=mock_open, read_data=b'\x1f\x8b')
def test_is_gz_file_true(mock_file):
    assert is_gz_file("dummy.gz") is True
    mock_file.assert_called_once_with("dummy.gz", 'rb')

@patch("builtins.open", new_callable=mock_open, read_data=b'\x00\x00')
def test_is_gz_file_false(mock_file):
    assert is_gz_file("dummy.txt") is False
    mock_file.assert_called_once_with("dummy.txt", 'rb')

@patch("builtins.open", side_effect=IOError("File not found"))
def test_is_gz_file_io_error(mock_file):
    with pytest.raises(IOError):
        is_gz_file("non_existent_file.gz")
    mock_file.assert_called_once_with("non_existent_file.gz", 'rb')


# --- Tests for get_mime_type ---
@patch('core_processing.magic.from_file')
def test_get_mime_type_pdf(mock_magic_from_file):
    mock_magic_from_file.return_value = "application/pdf"
    result = get_mime_type("dummy.pdf")
    assert result == "application/pdf"
    mock_magic_from_file.assert_called_once_with("dummy.pdf", mime=True)

@patch('core_processing.magic.from_file')
def test_get_mime_type_txt(mock_magic_from_file):
    mock_magic_from_file.return_value = "text/plain"
    result = get_mime_type("dummy.txt")
    assert result == "text/plain"
    mock_magic_from_file.assert_called_once_with("dummy.txt", mime=True)


# --- Tests for load_by_unstructured ---
@patch('core_processing.UnstructuredLoader')
def test_load_by_unstructured_success(MockUnstructuredLoader):
    mock_loader_instance = MagicMock()
    mock_docs = [MagicMock(spec=LangchainDocument)]
    mock_loader_instance.load.return_value = mock_docs
    MockUnstructuredLoader.return_value = mock_loader_instance

    mock_temp_file = MagicMock(spec=tempfile._TemporaryFileWrapper)
    mock_temp_file.name = "temp_file_path.txt"

    result = load_by_unstructured(mock_temp_file)

    MockUnstructuredLoader.assert_called_once_with(
        file_path="temp_file_path.txt",
        post_processors=[pytest.approx(lambda x: x is core_processing.clean_extra_whitespace)], # Check for function object
        chunking_strategy="basic",
        max_characters=10000000,
        include_orig_elements=False
    )
    mock_loader_instance.load.assert_called_once()
    assert result == mock_docs


# --- Tests for get_doc_id ---
def test_get_doc_id_generates_md5_hash():
    mock_doc = LangchainDocument(page_content="content", metadata={'source': 'test_source_path'})
    
    # MD5 hash of 'test_source_path' is c0923cf990a7079f3557690101390b3c
    expected_id = "c0923cf990a7" 
    
    result_id = get_doc_id(mock_doc)
    
    assert isinstance(result_id, str)
    assert len(result_id) == 12
    assert result_id == expected_id

def test_get_doc_id_no_source():
    mock_doc = LangchainDocument(page_content="content", metadata={}) # No source
    # MD5 hash of empty string "" is d41d8cd98f00b204e9800998ecf8427e
    expected_id = "d41d8cd98f00"
    result_id = get_doc_id(mock_doc)
    assert result_id == expected_id

def test_get_doc_id_non_string_source():
    mock_doc = LangchainDocument(page_content="content", metadata={'source': 123})
    # MD5 hash of "123" is "202cb962ac59075b964b07152d234b70"
    expected_id = "202cb962ac59"
    result_id = get_doc_id(mock_doc)
    assert result_id == expected_id


# --- Tests for split ---
@patch('core_processing.get_doc_id', return_value="fixed_doc_id")
def test_split_content_smaller_than_chunk_size(mock_get_id):
    mock_doc = LangchainDocument(page_content="Short content.", metadata={'source': 'test_source'})
    q_chunk_size = 100
    q_chunk_overlap = 10
    
    result_items = split(mock_doc, q_chunk_size, q_chunk_overlap)
    
    assert len(result_items) == 1
    assert result_items[0].content == "Short content."
    assert result_items[0].metadata['id'] == "fixed_doc_id_chunk_0"
    assert result_items[0].metadata['chunk_index'] == 0
    mock_get_id.assert_called_once_with(mock_doc)

@patch('core_processing.get_doc_id', return_value="fixed_doc_id")
@patch('core_processing.RecursiveCharacterTextSplitter')
def test_split_content_larger_than_chunk_size(MockTextSplitter, mock_get_id):
    mock_splitter_instance = MagicMock()
    predefined_chunks = ["This is chunk one.", "This is chunk two."]
    mock_splitter_instance.split_text.return_value = predefined_chunks
    MockTextSplitter.return_value = mock_splitter_instance

    long_content = "This is a long piece of text that will definitely be split into multiple chunks by the splitter."
    mock_doc = LangchainDocument(page_content=long_content, metadata={'source': 'long_doc_source'})
    q_chunk_size = 50
    q_chunk_overlap = 5

    result_items = split(mock_doc, q_chunk_size, q_chunk_overlap)

    MockTextSplitter.assert_called_once_with(
        chunk_size=q_chunk_size,
        chunk_overlap=q_chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    mock_splitter_instance.split_text.assert_called_once_with(long_content)
    mock_get_id.assert_called_once_with(mock_doc)
    
    assert len(result_items) == len(predefined_chunks)
    for i, item in enumerate(result_items):
        assert item.content == predefined_chunks[i]
        assert item.metadata['id'] == f"fixed_doc_id_chunk_{i}"
        assert item.metadata['chunk_index'] == i
        assert item.metadata['source'] == 'long_doc_source'

@patch('core_processing.get_doc_id', return_value="fixed_doc_id")
def test_split_empty_content(mock_get_id):
    mock_doc = LangchainDocument(page_content="", metadata={'source': 'empty_source'})
    q_chunk_size = 100
    q_chunk_overlap = 10
    
    result_items = split(mock_doc, q_chunk_size, q_chunk_overlap)
    assert len(result_items) == 0
    mock_get_id.assert_not_called() # get_doc_id is called inside the loop or for the single chunk case if page_content exists

# --- Tests for load ---
# Mocking core_processing.settings for specific tests if needed
# For example, to test delete=True vs delete=False for temp files

@patch('core_processing.is_gz_file', return_value=False)
@patch('core_processing.get_mime_type', return_value="text/plain")
@patch('core_processing.load_by_unstructured')
def test_load_normal_file(mock_load_by_unstructured, mock_get_mime, mock_is_gz):
    mock_docs = [LangchainDocument(page_content="doc content")]
    mock_load_by_unstructured.return_value = mock_docs
    
    mock_temp_file = MagicMock(spec=tempfile._TemporaryFileWrapper)
    mock_temp_file.name = "normal_file.txt"

    docs_result, mime_result = load(mock_temp_file)

    mock_is_gz.assert_called_once_with("normal_file.txt")
    mock_get_mime.assert_called_once_with("normal_file.txt") # Called once for non-gz path
    mock_load_by_unstructured.assert_called_once_with(mock_temp_file)
    assert docs_result == mock_docs
    assert mime_result == "text/plain"

@patch('core_processing.is_gz_file', return_value=True)
@patch('core_processing.gzip.open', new_callable=mock_open, read_data=b"decompressed_content")
@patch('core_processing.tempfile.NamedTemporaryFile')
@patch('core_processing.get_mime_type')
@patch('core_processing.load_by_unstructured')
@patch('core_processing.os.remove') # To mock os.remove for the decompressed temp file
@patch('core_processing.settings') # To control delete_temp_file behavior
def test_load_gz_file(
    mock_settings, mock_os_remove, mock_load_by_unstructured, 
    mock_get_mime_type_on_decompressed, MockNamedTemporaryFile, 
    mock_gzip_open, mock_is_gz_file
):
    # Configure settings mock
    mock_settings.delete_temp_file = True

    # Mock for the decompressed temporary file
    mock_decompressed_file_obj = MagicMock(spec=tempfile._TemporaryFileWrapper)
    mock_decompressed_file_obj.name = "decompressed_temp.file"
    MockNamedTemporaryFile.return_value.__enter__.return_value = mock_decompressed_file_obj

    # Mock for load_by_unstructured
    mock_final_docs = [LangchainDocument(page_content="final content")]
    mock_load_by_unstructured.return_value = mock_final_docs
    
    # Mock for get_mime_type on the decompressed file
    mock_get_mime_type_on_decompressed.return_value = "text/plain-decompressed"

    # Original gzipped temp file
    mock_original_temp_file = MagicMock(spec=tempfile._TemporaryFileWrapper)
    mock_original_temp_file.name = "original.gz"

    docs_result, mime_result = load(mock_original_temp_file)

    mock_is_gz_file.assert_called_once_with("original.gz")
    mock_gzip_open.assert_called_once_with("original.gz", 'rb')
    MockNamedTemporaryFile.assert_called_once_with(mode='wb', delete=False, suffix=".gz_decompressed")
    
    # Check if decompressed_file_obj.write was called (indirectly via shutil.copyfileobj or loop)
    # This part is tricky as the current `load` function uses a loop.
    # We'll check if the decompressed file name was used for loading.
    
    # Assert load_by_unstructured was called with a wrapper around the decompressed path
    assert mock_load_by_unstructured.call_args is not None
    args, _ = mock_load_by_unstructured.call_args
    assert args[0].name == "decompressed_temp.file"

    mock_get_mime_type_on_decompressed.assert_called_once_with("decompressed_temp.file")
    
    assert docs_result == mock_final_docs
    assert mime_result == "text/plain-decompressed"

    # Check if os.remove was called on the decompressed temp file if settings.delete_temp_file is True
    if mock_settings.delete_temp_file:
        mock_os_remove.assert_called_once_with("decompressed_temp.file")
    else:
        mock_os_remove.assert_not_called()

@patch('core_processing.is_gz_file', return_value=False)
@patch('core_processing.get_mime_type', return_value="application/pdf")
@patch('core_processing.load_by_unstructured')
def test_load_pdf_uses_unstructured_directly(mock_load_by_unstructured, mock_get_mime, mock_is_gz):
    mock_docs = [LangchainDocument(page_content="pdf content")]
    mock_load_by_unstructured.return_value = mock_docs
    
    mock_temp_file = MagicMock(spec=tempfile._TemporaryFileWrapper)
    mock_temp_file.name = "document.pdf"

    docs_result, mime_result = load(mock_temp_file)

    mock_is_gz.assert_called_once_with("document.pdf")
    mock_get_mime.assert_called_once_with("document.pdf")
    mock_load_by_unstructured.assert_called_once_with(mock_temp_file)
    assert docs_result == mock_docs
    assert mime_result == "application/pdf"

# Example of how to run tests with pytest (if this file is executed directly)
if __name__ == "__main__":
    pytest.main()
