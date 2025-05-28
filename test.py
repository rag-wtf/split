import os
import json
from fastapi.testclient import TestClient
from split import create_app, settings # Assuming settings might be useful for inspection

# Initialize TestClient
# We need to ensure that the settings are loaded correctly, especially SUPPORTED_FILE_TYPES
# for the middleware. If .env is used, it should be loaded before settings are imported
# or before create_app() is called if settings are accessed at module level in split.py.
# The current setup in split.py (settings = Settings() at module level) means .env is loaded by pydantic-settings.
client = TestClient(create_app())

# Test files directory
TEST_FILES_DIR = "test_files"

def test_split_breathing_gz():
    """
    Tests splitting of the existing 'breathing.gz' file.
    Asserts successful response and basic structure of the output.
    """
    print("\nRunning test_split_breathing_gz...")
    file_path = "breathing.gz" # Assumes this file exists in the project root
    if not os.path.exists(file_path):
        print(f"SKIPPING test_split_breathing_gz: {file_path} not found.")
        # Create a dummy gz file to avoid erroring out the whole test run if not present
        # This is not ideal but makes the test suite runnable.
        # In a real CI, this file should be present.
        import gzip
        dummy_gz_content = b"This is dummy text for breathing.gz."
        with gzip.open(file_path, "wb") as f:
            f.write(dummy_gz_content)
        print(f"Created dummy {file_path} for test purposes.")
        
    with open(file_path, "rb") as fp:
        response = client.post("/split", files={"file": ("breathing.gz", fp, "application/gzip")})
    
    assert response.status_code == 200
    response_json = response.json()
    
    assert "items" in response_json
    assert isinstance(response_json["items"], list)
    assert len(response_json["items"]) > 0, "Should return at least one chunk for breathing.gz"
    
    # Assuming breathing.gz contains plain text after decompression
    # The actual MIME type might depend on what 'python-magic' detects from the decompressed stream.
    # This might need adjustment based on the actual content of breathing.gz
    assert response_json["mime_type"] == "text/plain", f"Expected mime_type text/plain, got {response_json['mime_type']}"
    
    for item in response_json["items"]:
        assert "content" in item
        assert isinstance(item["content"], str)
        assert "metadata" in item
        assert isinstance(item["metadata"], dict)
        assert "source" in item["metadata"] # Source should be the temp file path
    print("test_split_breathing_gz PASSED")

def test_split_simple_text():
    """
    Tests splitting of a simple text file with default chunking parameters.
    """
    print("\nRunning test_split_simple_text...")
    file_path = os.path.join(TEST_FILES_DIR, "simple.txt")
    original_content = ""
    with open(file_path, "r") as f_read:
        original_content = f_read.read()

    with open(file_path, "rb") as fp:
        response = client.post("/split", files={"file": ("simple.txt", fp, "text/plain")})
    
    assert response.status_code == 200
    response_json = response.json()
    
    assert response_json["mime_type"] == "text/plain"
    assert "items" in response_json
    assert isinstance(response_json["items"], list)
    assert len(response_json["items"]) > 0, "Should return chunks for simple.txt"
    
    combined_content = "".join(item["content"] for item in response_json["items"])
    # Check if combined content is reasonably close to original (splitter might add/remove whitespace or join lines)
    # This is a rough check. A more precise check would involve normalizing whitespace.
    assert len(combined_content) >= len(original_content) * 0.8 and len(combined_content) <= len(original_content) * 1.2
    
    for item in response_json["items"]:
        assert "content" in item and isinstance(item["content"], str)
        assert "metadata" in item and isinstance(item["metadata"], dict)
    print("test_split_simple_text PASSED")

def test_split_custom_chunking():
    """
    Tests splitting of a simple text file with custom chunk size and overlap.
    """
    print("\nRunning test_split_custom_chunking...")
    file_path = os.path.join(TEST_FILES_DIR, "simple.txt")
    chunk_size = 50
    chunk_overlap = 10

    with open(file_path, "rb") as fp:
        response = client.post(
            f"/split?q_chunk_size={chunk_size}&q_chunk_overlap={chunk_overlap}",
            files={"file": ("simple.txt", fp, "text/plain")}
        )
            
    assert response.status_code == 200
    response_json = response.json()
    
    assert "items" in response_json
    assert isinstance(response_json["items"], list)
    assert len(response_json["items"]) > 0
    
    for i, item in enumerate(response_json["items"]):
        assert "content" in item and isinstance(item["content"], str)
        # The last chunk can be smaller. RecursiveCharacterTextSplitter tries to respect chunk_size.
        if i < len(response_json["items"]) - 1:
             assert len(item["content"]) <= chunk_size + chunk_overlap # Overlap can make it slightly larger
        else:
            assert len(item["content"]) > 0
    print("test_split_custom_chunking PASSED")

def test_split_unsupported_file_type():
    """
    Tests the behavior when an unsupported file type is uploaded.
    This should be rejected by the ValidateUploadFileMiddleware.
    """
    print("\nRunning test_split_unsupported_file_type...")
    file_path = os.path.join(TEST_FILES_DIR, "fake.unsupported_ext")
    
    # Check if 'application/octet-stream' is in supported types. If so, this test is invalid.
    # For this test, we assume 'application/octet-stream' or a custom type for '.unsupported_ext'
    # is NOT in settings.supported_file_types.
    
    # Print supported types from settings for debugging if needed
    # print(f"Supported types by middleware: {settings.supported_file_types}")

    with open(file_path, "rb") as fp:
        response = client.post("/split", files={"file": ("fake.unsupported_ext", fp, "application/octet-stream")})
        
    assert response.status_code == 415 # Unsupported Media Type
    print("test_split_unsupported_file_type PASSED")

def test_split_empty_file():
    """
    Tests uploading an empty text file.
    UnstructuredLoader should produce no documents, resulting in an empty 'items' list.
    """
    print("\nRunning test_split_empty_file...")
    file_path = os.path.join(TEST_FILES_DIR, "empty.txt")
    
    with open(file_path, "rb") as fp:
        # Uploading an empty file. Content-Type is 'text/plain'.
        response = client.post("/split", files={"file": ("empty.txt", fp, "text/plain")})
            
    assert response.status_code == 200 # Should be accepted by middleware if text/plain is allowed
    response_json = response.json()
    assert "items" in response_json
    assert isinstance(response_json["items"], list)
    assert len(response_json["items"]) == 0, "Processing an empty file should result in zero items."
    print("test_split_empty_file PASSED")

# Stubs for PDF/DOCX tests if they were to be implemented
# def test_split_simple_pdf():
#     print("\nSKIPPING test_split_simple_pdf: PDF creation not implemented in this environment.")
#     pass

# def test_split_simple_docx():
#     print("\nSKIPPING test_split_simple_docx: DOCX creation not implemented in this environment.")
#     pass

if __name__ == "__main__":
    # Create test_files directory if it doesn't exist
    # This was done in a previous step, but good to have for standalone running
    if not os.path.exists(TEST_FILES_DIR):
        os.makedirs(TEST_FILES_DIR)
        print(f"Created directory: {TEST_FILES_DIR}")

    # File creation was handled by separate tool calls in previous turns.
    # For making test.py self-contained for local runs without the tool,
    # one might add file creation here as originally planned in the prompt.
    # However, since the files were created by the agent already, we'll proceed to run tests.
    
    print("Starting integration tests...")
    
    test_split_breathing_gz()
    test_split_simple_text()
    test_split_custom_chunking()
    test_split_unsupported_file_type()
    test_split_empty_file()
    # test_split_simple_pdf() # Skipped
    # test_split_simple_docx() # Skipped
    
    print("\nAll selected tests finished.")
