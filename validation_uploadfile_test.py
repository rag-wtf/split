import pytest
from fastapi import FastAPI, status, Request
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from validation_uploadfile import ValidateUploadFileMiddleware, FileTypeName  # Assuming app is the FastAPI instance

@pytest.fixture
def app():
    app = FastAPI()

    # Add middleware with different configurations for testing
    app.add_middleware(
        ValidateUploadFileMiddleware,
        app_paths=["/upload"],
        max_size=1024,  # 1KB for testing purposes
        file_types=[FileTypeName.JPEG, FileTypeName.PNG]
    )

    @app.post("/upload")
    async def upload_file(request: Request):
        return {"detail": "File uploaded successfully"}

    return app

@pytest.fixture
async def async_client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        yield client

pytestmark = pytest.mark.anyio

async def test_valid_file_upload(async_client):
    # Valid JPEG file upload
    files = {"file": ("test.jpg", b"fake-jpeg-data", "image/jpeg")}
    response = await async_client.post("/upload", files=files)
    assert response.status_code == status.HTTP_200_OK


async def test_invalid_content_type(async_client):
    # Upload a file with an unsupported content type
    files = {"file": ("test.txt", b"fake-text-data", "text/plain")}
    response = await async_client.post("/upload", files=files)
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

async def test_large_file_upload(async_client):
    # Upload a file larger than the allowed size
    large_file_data = b"fake-large-data" * 1024  # Creating a file larger than 1KB
    files = {"file": ("large.jpg", large_file_data, "image/jpeg")}
    response = await async_client.post("/upload", files=files)
    assert response.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE

async def test_no_files_provided(async_client):
    # Test case where no files are provided
    response = await async_client.post("/upload")
    assert response.status_code == status.HTTP_400_BAD_REQUEST


async def test_multiple_files_upload(async_client):
    # Upload multiple files
    files = [
        ("file1", ("test1.jpg", b"fake-jpeg-data", "image/jpeg")),
        ("file2", ("test2.png", b"fake-png-data", "image/png"))
    ]
    response = await async_client.post("/upload", files=files)
    assert response.status_code == status.HTTP_200_OK


async def test_edge_case_invalid_request_no_file(async_client):
    headers = {"content-type": "application/json"}  
    response = await async_client.post("/upload", headers=headers)
    assert response.status_code == status.HTTP_400_BAD_REQUEST

async def test_edge_case_invalid_request_incorrect_content_type(async_client):
    files = {"file": ("test.json", b'{"content-type": "application/json"}', "application/json")}    
    response = await async_client.post("/upload", files=files)
    assert response.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
