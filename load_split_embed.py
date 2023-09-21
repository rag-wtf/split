from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette_validation_uploadfile import ValidateUploadFileMiddleware
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from open.text.embeddings.openai import OpenAIEmbeddings
from typing import List
import tempfile
import os
import hashlib

router = APIRouter()


def create_app():
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
        app_path="/load_split_embed",
        max_size=int(os.getenv("MAX_FILE_SIZE_IN_MB")) * 1048576,  # 1 MB
        file_type=os.getenv("SUPPORTED_FILE_TYPES").split(",")
    )

    app.include_router(router)

    return app


def load(file):
    # REF: https://python.langchain.com/docs/integrations/document_loaders/unstructured_file
    loader = UnstructuredFileLoader(
        file_path=file.name,
        post_processors=[clean_extra_whitespace],
    )
    return loader.load()


md5 = hashlib.md5()


def get_doc_id(doc):
    md5.update(doc.metadata['source'].encode('utf-8'))
    uid = md5.hexdigest()[:12]
    return uid


def split(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
        length_function=len)
    chunks = text_splitter.split_text(doc.page_content)
    return chunks


def embed(texts):
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("EMBEDDING_MODEL_KEY"),
        openai_api_base=os.getenv("EMBEDDING_MODEL_URL")
    )
    results = embeddings.embed_documents(texts)
    return results


class LoadSplitEmbedResponse(BaseModel):
    id: str
    text: str
    embedding: List[float]


@router.post("/load_split_embed")
async def load_split_embed(file: UploadFile = File(...)):
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
        docs = load(temp)
        doc = docs[0]
        texts = split(doc)
        print(len(texts))
        print(texts[1])
        embeddings = embed(texts)
        print(len(embeddings))
        print(embeddings[0])
        id = get_doc_id(doc)
        responses = []
        for i, text in enumerate(texts):
            responses.append(
                LoadSplitEmbedResponse(
                    id=f'{id}-{i}',
                    text=text,
                    embedding=embeddings[i]
                )
            )
        return responses


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        create_app(), host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", 8000))
    )
elif os.getenv("RUNTIME") == "aws-lambda":
    from mangum import Mangum
    handler = Mangum(create_app())
