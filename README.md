# ✂️ split

## 🎯 Project Purpose and ⚙️ Architecture

### Purpose

This project provides an API endpoint for loading various document types (e.g., PDF, DOCX, HTML, TXT), splitting their textual content into manageable chunks, and returning these chunks along with metadata. This specific service focuses on the **Load** and **Split** stages. The output is designed to be suitable for preprocessing data for downstream tasks, particularly for Retrieval Augmented Generation (RAG) pipelines where these chunks would typically be fed into an embedding model.

### Architecture

The service is built as a Python [FastAPI](https://fastapi.tiangolo.com/) application. It leverages the [Unstructured](https://unstructured.io/) library for robust document parsing and content extraction, and [Langchain](https://www.langchain.com/) for its text splitting capabilities (specifically `RecursiveCharacterTextSplitter`).

The application is designed to be:

  - **📦 Containerized:** Using Docker for consistent environments and deployment.
  - **☁️ Serverless-ready:** Deployable as an AWS Lambda function, managed via the Serverless Framework.

-----

## 🔑 Environment Variables

The application uses several environment variables for configuration, managed through a `.env` file and a `config.py` file.

| Variable | Purpose | Default (in `config.py`) | File(s) Used In |
| --- | --- | --- | --- |
| `DELETE_TEMP_FILE` | If `1`, temporary files created during processing will be deleted. | `True` | `config.py`, `split.py` |
| `NLTK_DATA` | Path to NLTK data directory, needed for tokenizers used by `unstructured`. | `/tmp/nltk_data` | `config.py`, `split.py` |
| `MAX_FILE_SIZE_IN_MB` | Maximum allowed file size for uploads in Megabytes. | `10.0` | `config.py`, `split.py` |
| `SUPPORTED_FILE_TYPES` | Comma-separated string of allowed MIME types for uploaded files. | See `config.py` for a comprehensive list | `config.py`, `split.py` |
| `CHUNK_SIZE` | Target size for text chunks in characters. | `500` | `config.py`, `split.py` |
| `CHUNK_OVERLAP` | Number of characters to overlap between consecutive chunks. | `20` | `config.py`, `split.py` |
| `HOST` | Host address for the Uvicorn server in local development. | `0.0.0.0` | `config.py` |
| `PORT` | Port for the Uvicorn server in local development. | `8000` | `config.py` |
| `RUNTIME` | Used to indicate the running environment, e.g., "aws-lambda". | `None` | `config.py`, `split.py` |
| `HF_HOME` | Path to HuggingFace cache directory. Relevant if `unstructured` uses models from HuggingFace Hub. | `/tmp/hf_home` | `config.py` |

-----

## 💻 Setup and Local Development

### ✅ Prerequisites

  - Python 3.11+
  - Docker
  - Node.js (for Serverless Framework)

### 🛠️ Steps

1.  **Clone the repository.**

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **NLTK Data:**
    The `unstructured` library requires NLTK data packages. The application is configured to look for them in the path specified by the `NLTK_DATA` environment variable.

4.  **Create a `.env` file:**
    Copy the contents of the example below into a `.env` file in the project root to configure the application for local development.

    ```env
    HOST=0.0.0.0
    PORT=8000
    DELETE_TEMP_FILE=1
    NLTK_DATA=/tmp/nltk_data
    MAX_FILE_SIZE_IN_MB=10
    SUPPORTED_FILE_TYPES=text/plain,application/pdf,text/html,text/markdown,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/epub+zip,message/rfc822,application/gzip
    CHUNK_SIZE=500
    CHUNK_OVERLAP=20
    HF_HOME=/tmp/hf_home
    ```

5.  **Running the application locally:**
    Use the provided shell script to start the server with Uvicorn:

    ```bash
    ./start_server.sh
    ```

    Alternatively, run the `split.py` script directly:

    ```bash
    python split.py
    ```

6.  **Running with Docker:**

      - **Build the Docker image:**
        ```bash
        ./docker-build.sh
        ```
      - **Run the Docker container:**
        ```bash
        ./docker-run.sh
        ```

    The `docker-compose.yaml` file is also available for running the service with Docker Compose.

-----

## 🔗 API Endpoints

### 1\. `POST /split`

Uploads a document, splits its textual content, and returns the chunks.

  - **Request:**
      - Method: `POST`
      - Content-Type: `multipart/form-data`
      - Body: Must include a `file` field containing the document.
  - **Query Parameters:**
      - `q_chunk_size` (integer, optional): Desired chunk size. Defaults to `CHUNK_SIZE`.
      - `q_chunk_overlap` (integer, optional): Desired chunk overlap. Defaults to `CHUNK_OVERLAP`.
  - **Response (200 OK):**
    A JSON object with the following structure:
    ```json
    {
      "content": "string or null",
      "mime_type": "string",
      "items": [
        {
          "content": "string",
          "metadata": {
            "source": "string",
            "id": "string",
            // ... other metadata
          }
        }
      ]
    }
    ```
  - **`curl` Example:**
    ```bash
    curl -X POST -F "file=@/path/to/your/document.pdf" "http://localhost:8000/split?q_chunk_size=1000&q_chunk_overlap=100"
    ```

### 2\. `GET /split/config`

Returns the current operational configuration of the service.

  - **Response (200 OK):**
    A JSON object detailing the service's settings:
    ```json
    {
      "delete_temp_file": true,
      "nltk_data": "/tmp/nltk_data",
      "max_file_size_in_mb": 10.0,
      "supported_file_types": [
        "text/plain",
        "application/pdf",
        // ...
      ],
      "chunk_size": 500,
      "chunk_overlap": 50
    }
    ```
  - **`curl` Example:**
    ```bash
    curl http://localhost:8000/split/config
    ```

-----

## 🚀 Deployment

### ☁️ AWS Lambda

The service is designed for serverless deployment on AWS Lambda using the Serverless Framework. The `serverless.yml` file configures the Lambda function, API Gateway trigger, and environment variables. The `Dockerfile-AwsLambda` is used to build the container image for deployment.

The `.github/workflows/dev.yml` file contains a GitHub Actions workflow for deploying to a development environment on AWS.

### 🖥️ VPS

A GitHub Actions workflow is also provided for deploying the application to a Virtual Private Server (VPS) in `.github/workflows/deploy-vps.yml`.

-----

## 📦 Dependency Management

The project uses multiple `requirements.txt` files for different environments:

  * **`requirements.txt`**: For local development and testing.
  * **`deploy-requirements.txt`**: Production dependencies for the full-featured AWS Lambda deployment.
  * **`requirements-text-only.txt`**: A minimal set of dependencies for a text-only version of the service.

It is important to regularly review and update dependencies and use tools like GitHub Dependabot, Snyk, or Trivy for vulnerability scanning.

-----

## 📜 License

This project is licensed under the MIT License.