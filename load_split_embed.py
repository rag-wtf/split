from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette_validation_uploadfile import ValidateUploadFileMiddleware
from fastapi.responses import StreamingResponse
import tempfile
import os
import random
import asyncio

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
    app_path="/upload",
    max_size=1048576,  # 1 MB
    file_type=["text/plain",
               "image/jpeg",
               "image/png",
               "video/mp4",
               "video/webm",
               "application/json",
               "application/pdf",
               ]
)


async def simulate_long_running_process():
    # simulate the progress of processing by sending 100 zero bytes to the client
    max_bytes = 100
    zero_bytes = bytes(max_bytes)
    progress = 0
    while progress < max_bytes:
        num_of_bytes = random.randint(5, 15)
        startIndex = progress
        progress += num_of_bytes
        endIndex = progress
        if (endIndex > max_bytes):
            endIndex = max_bytes
        print(startIndex, endIndex - 1)
        sleeping_time_ms = random.randint(1, 50)
        sleeping_time_s = sleeping_time_ms * 0.01
        await asyncio.sleep(sleeping_time_s)
        yield zero_bytes[startIndex: endIndex]


@app.post("/upload")
async def create_upload_file(file: UploadFile = File(...)):
    chunk_size = 1024 * 1024  # 1 MB

    with tempfile.NamedTemporaryFile(
            mode='wb',
            buffering=chunk_size,
            delete=os.getenv("DELETE", True)) as temp:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            temp.write(chunk)
        temp.flush()
    print("Upload completed!")
    return StreamingResponse(simulate_long_running_process())


def test_upload_file():
    from fastapi.testclient import TestClient
    client = TestClient(app)
    with open("test.json", "rb") as file:
        response = client.post("/upload", files={"file": file})
    # iterate over the response content
    for chunk in response.iter_bytes():
        print(f"progress {chunk}")


if __name__ == "__main__":
    # test_upload_file()
    import uvicorn
    uvicorn.run(
        app, host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", 8000))
    )
elif os.getenv("RUNTIME") == "aws-lambda":
    from mangum import Mangum
    handler = Mangum(app)
