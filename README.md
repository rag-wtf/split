# load-split-embed

## Project Purpose and Architecture

### Purpose

This project provides an API endpoint for loading various document types (e.g., PDF, DOCX, HTML, TXT), splitting their textual content into manageable chunks, and returning these chunks along with metadata. While the project name includes "embed," this specific service focuses on the **Load** and **Split** stages. The output is designed to be suitable for preprocessing data for downstream tasks, particularly for Retrieval Augmented Generation (RAG) pipelines where these chunks would typically be fed into an embedding model.

### Architecture

The service is built as a Python [FastAPI](https://fastapi.tiangolo.com/) application. It leverages the [Unstructured](https://unstructured.io/) library for robust document parsing and content extraction, and [Langchain](https://www.langchain.com/) for its text splitting capabilities (specifically `RecursiveCharacterTextSplitter`).

The application is designed to be:
-   **Containerized:** Using Docker for consistent environments and deployment.
-   **Serverless-ready:** Deployable as an AWS Lambda function, managed via the Serverless Framework.

## Environment Variables

The application uses several environment variables for configuration. For local development, these can be set in a `.env` file.

| Variable                 | Purpose                                                                                                | Default (in code if not set) / Example Value | File(s) Used In        |
| ------------------------ | ------------------------------------------------------------------------------------------------------ | -------------------------------------------- | ---------------------- |
| `HOST`                   | Host address for the Uvicorn server in local development.                                              | `0.0.0.0`                                    | `dev.py`               |
| `PORT`                   | Port for the Uvicorn server in local development.                                                      | `8000`                                       | `dev.py`               |
| `DELETE_TEMP_FILE`       | If `True`, temporary files created during processing will be deleted. Recommended: `True`.             | `False` (if env var is empty/not set)        | `split.py`             |
| `NLTK_DATA`              | Path to NLTK data directory. Needed for `nltk` tokenizers (e.g., `punkt`) used by `unstructured`.       | `./nltk_data` (example)                      | `split.py`             |
| `MAX_FILE_SIZE_IN_MB`    | Maximum allowed file size for uploads in Megabytes.                                                    | `10.0`                                       | `split.py`             |
| `SUPPORTED_FILE_TYPES`   | Comma-separated string of allowed MIME types for uploaded files.                                       | See `.env.example` for a common list         | `split.py`             |
| `CHUNK_SIZE`             | Target size for text chunks in characters.                                                             | `500`                                        | `split.py`, `serverless.yml` |
| `CHUNK_OVERLAP`          | Number of characters to overlap between consecutive chunks.                                            | `50`                                         | `split.py`, `serverless.yml` |
| `RUNTIME`                | Used to indicate if running in a specific environment, e.g., "aws-lambda" for AWS Lambda context.      | Not set by default                           | `split.py`             |
| `HF_HOME`                | Path to HuggingFace cache directory. Relevant if `unstructured` uses models from HuggingFace Hub.        | `/tmp/hf_home` (in `serverless.yml`)         | `serverless.yml`       |

## Setup and Local Development

### Prerequisites

-   Python 3.11
-   Docker (for containerized development/deployment)
-   Access to a terminal or command prompt.

### Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **NLTK Data:**
    The `unstructured` library, a core dependency, often relies on `nltk` for text processing, which in turn requires specific data packages (like the `punkt` tokenizer and `averaged_perceptron_tagger`). The `NLTK_DATA` environment variable tells `nltk` where to look for these packages.
    You might need to download these manually or via a script if they are not present in your `NLTK_DATA` directory:
    ```python
    import nltk
    nltk.download('punkt', download_dir='./nltk_data')
    nltk.download('averaged_perceptron_tagger', download_dir='./nltk_data')
    # Ensure NLTK_DATA environment variable is set to ./nltk_data or your chosen path.
    ```
    It's recommended to download these into a local directory (e.g., `nltk_data` in the project root) and point the `NLTK_DATA` environment variable to it.

4.  **Create `.env` file:**
    Create a file named `.env` in the project root for local environment variables. You can copy `.env.example` (see below) to `.env` and modify as needed.

    **.env.example:**
    ```env
    HOST=0.0.0.0
    PORT=8000
    DELETE_TEMP_FILE=True
    NLTK_DATA=./nltk_data # Or a suitable path for your nltk data
    MAX_FILE_SIZE_IN_MB=10
    SUPPORTED_FILE_TYPES=text/plain,application/pdf,text/html,text/markdown,application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation,application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/epub+zip,message/rfc822,application/gzip
    CHUNK_SIZE=500
    CHUNK_OVERLAP=50
    # RUNTIME=local # Optional: can be used to signal local runtime if needed
    # HF_HOME=./hf_cache # Optional: if you want to control HuggingFace cache location locally
    ```

5.  **Running the application locally (Uvicorn):**
    You can run the FastAPI application directly using Uvicorn:
    ```bash
    python dev.py
    ```
    Alternatively, you can run:
    ```bash
    uvicorn split:app --reload --host ${HOST:-0.0.0.0} --port ${PORT:-8000}
    ```
    The application will typically be available at `http://localhost:8000` (or the host/port specified in your `.env` file or command).

6.  **Running with Docker (local):**
    The repository includes helper scripts for Docker operations.
    -   **Build the Docker image:**
        ```bash
        ./docker-build.sh 
        # Equivalent to: docker build -t load-split-service .
        ```
    -   **Run the Docker container:**
        Make sure you have a `.env` file in the project root, as the `docker-run.sh` script (or the command below) will pass it to the container.
        ```bash
        ./docker-run.sh
        # Equivalent to: docker run -it --rm -p 8000:8000 --env-file .env load-split-service:latest
        ```
        The service inside the container will be accessible on port 8000 of your host machine.

## API Endpoints

The service provides the following API endpoints:

### 1. `POST /split`

Uploads a document, splits its textual content into chunks, and returns the chunks.

-   **Request:**
    -   Method: `POST`
    -   Content-Type: `multipart/form-data`
    -   Body: Must include a `file` field containing the document to process.
-   **Query Parameters:**
    -   `q_chunk_size` (integer, optional): Desired chunk size in characters. Defaults to the value from the `CHUNK_SIZE` environment variable.
    -   `q_chunk_overlap` (integer, optional): Desired chunk overlap in characters. Defaults to the value from the `CHUNK_OVERLAP` environment variable.
-   **Response (Success: 200 OK):**
    A JSON object with the following structure (based on Pydantic models `DocumentResponse` and `Chunk` in `split.py`):
    ```json
    {
      "content": null, // Original content is not returned by default
      "mime_type": "string", // Detected MIME type of the uploaded file
      "items": [
        {
          "content": "string", // Text content of a specific chunk
          "metadata": {
            "source": "string", // Path to the temporary file used for processing
            "id": "string",     // MD5 hash of the chunk content
            // ... other metadata extracted by Unstructured (e.g., page_number, filename)
          }
        }
        // ... more items (chunks)
      ]
    }
    ```
-   **`curl` Example:**
    ```bash
    curl -X POST -F "file=@/path/to/your/document.pdf" "http://localhost:8000/split?q_chunk_size=1000&q_chunk_overlap=100"
    ```

### 2. `GET /split/config`

Returns the current operational configuration of the service.

-   **Response (Success: 200 OK):**
    A JSON object detailing the service's settings (based on Pydantic model `SplitConfig` in `split.py`):
    ```json
    {
      "delete_temp_file": true,
      "nltk_data": "/path/to/nltk_data",
      "max_file_size_in_mb": 10.0,
      "supported_file_types": [
        "text/plain",
        "application/pdf",
        // ... other configured MIME types
      ],
      "chunk_size": 500,
      "chunk_overlap": 50
    }
    ```
-   **`curl` Example:**
    ```bash
    curl http://localhost:8000/split/config
    ```

## Deployment (AWS Lambda)

This service is designed for serverless deployment on AWS Lambda using the Serverless Framework.

-   **Configuration:** The `serverless.yml` file in the project root defines the AWS Lambda function, API Gateway trigger, environment variables, and other deployment settings.
-   **Container Image:** The `Dockerfile-AwsLambda` is used to build the Docker container image that will be deployed to AWS Lambda.
-   **Deployment Steps (High-Level):**
    1.  **Configure AWS Credentials:** Ensure your AWS CLI is configured with credentials that have permissions to deploy Lambda functions, API Gateway, ECR (for container images), etc.
    2.  **Install Serverless Framework:** If you haven't already, install the Serverless Framework: `npm install -g serverless`.
    3.  **Deploy:** Navigate to the project root directory and run:
        ```bash
        serverless deploy
        ```
        This command will package the application, build the Docker image using `Dockerfile-AwsLambda`, push it to Amazon ECR, and deploy the Lambda function and API Gateway endpoint as defined in `serverless.yml`.

Refer to the Serverless Framework documentation and AWS documentation for more detailed deployment guides.

## Dependency Management

This project uses several `requirements.txt` files to manage Python dependencies for different environments and purposes:

*   **`requirements.txt`**: Used for local development and running tests. It includes all production dependencies from `deploy-requirements.txt` plus additional tools needed for development (e.g., `pytest`, `httpx`).
*   **`deploy-requirements.txt`**: Specifies the exact Python dependencies required for production deployments, particularly for the AWS Lambda environment (used by `Dockerfile-AwsLambda`). This list is optimized for running all features of the service.
*   **`requirements-text-only.txt`**: A subset of dependencies for a minimal, text-only version of the service (used by `Dockerfile-Text-Only`). This helps reduce package size if only plain text processing is needed.

When adding or updating core dependencies, ensure consistency across relevant files, primarily `deploy-requirements.txt` and then `requirements.txt`. For the text-only version, evaluate if the dependency is applicable.

### Vulnerability Scanning and Updates

To maintain the security and stability of the application, it's important to manage dependencies proactively:

*   **Regular Reviews:** Periodically review dependencies to ensure they are necessary and up-to-date.
*   **Security Bulletins:** Monitor security advisories for key libraries like FastAPI, Langchain, Unstructured, and their underlying parsing libraries.
*   **Automated Scanning:** Implement automated vulnerability scanning. Recommended tools include:
    *   **GitHub Dependabot:** Enable for automated alerts and pull requests for security updates (if the project is hosted on GitHub).
    *   Tools like **Snyk** or **Trivy** can be integrated into CI/CD pipelines for deeper scans of dependencies and Docker images.
*   **Update Policy:** Apply updates, especially security patches, promptly after thorough testing in a non-production environment.
