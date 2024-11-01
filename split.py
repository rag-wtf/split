from fastapi import FastAPI, APIRouter, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from validation_uploadfile import ValidateUploadFileMiddleware
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyMuPDFLoader
from unstructured.cleaners.core import clean_extra_whitespace
from langchain.text_splitter import RecursiveCharacterTextSplitter

from pydantic import BaseModel
from typing import List, Mapping, Any, Optional
import tempfile
import os
import hashlib
import nltk
import gzip
import magic

delete_temp_file = bool(os.getenv("DELETE_TEMP_FILE", ""))
nltk_data = os.getenv("NLTK_DATA")
max_file_size_in_mb = float(os.getenv("MAX_FILE_SIZE_IN_MB"))
supported_file_types = os.getenv("SUPPORTED_FILE_TYPES")
chunk_size = int(os.getenv("CHUNK_SIZE"))
chunk_overlap = int(os.getenv("CHUNK_OVERLAP"))

print("delete_temp_file:", delete_temp_file)
print("nltk_data:", nltk_data)
print("max_file_size_in_mb:", max_file_size_in_mb)
print("supported_file_types:", supported_file_types)
print("chunk_size:", chunk_size)
print("chunk_overlap:", chunk_overlap)

# Set the nltk.data.path with environment variable
nltk.data.path.append(nltk_data)

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
        app_paths="/split",
        max_size=max_file_size_in_mb * 1048576,  # 1 MB
        file_types=supported_file_types.split(",")
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    app.include_router(router)

    return app


def is_gz_file(file_path):
    with open(file_path, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'


def get_mime_type(file_path):
    mime_type = magic.from_file(file_path, mime=True)
    return mime_type


def load(file):
    # REF: https://python.langchain.com/docs/integrations/document_loaders/unstructured_file
    if is_gz_file(file.name):
        print(f'The {file.name} is gzip compressed.')
        with gzip.open(file.name, 'rb') as gzipped_file:
            with tempfile.NamedTemporaryFile(
                    mode='wb',
                    delete=delete_temp_file) as decompressed_file:
                for chunk in gzipped_file:
                    decompressed_file.write(chunk)
                decompressed_file.flush()
                docs = load_by_unstructured(decompressed_file)
                return docs, get_mime_type(decompressed_file.name)
    else:
        mime_type = get_mime_type(file.name)
        if mime_type == 'application/pdf':
            loader = PyMuPDFLoader(file.name, extract_images=True)
            try:
                docs = loader.load()
                if len(docs) == 0:
                    docs = load_by_unstructured(file)
            except:
                docs = load_by_unstructured(file)

        else:
            docs = load_by_unstructured(file)    
        return docs, mime_type


def load_by_unstructured(file):
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


def split(doc, q_chunk_size, q_chunk_overlap):
    items = []
    if (len(doc.page_content) > q_chunk_size):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=q_chunk_size,
            chunk_overlap=q_chunk_overlap,
            length_function=len
        )
        texts = text_splitter.split_text(doc.page_content)
        print(len(texts))
        print(texts[1])
        id = get_doc_id(doc)
        doc.metadata['id'] = id
        for i, text in enumerate(texts):
            items.append(
                DocumentItem(
                    content=text,
                    metadata=doc.metadata,
                )
            )
    else:
        print(f'len(doc.page_content) {len(doc.page_content)} <= chunk_size {q_chunk_size}: doc.page_content:',
            doc.page_content)
    return items


class DocumentItem(BaseModel):
    content: str
    metadata: Mapping[str, Any]


class Document(BaseModel):
    content: Optional[str]
    mime_type: str
    items: List[DocumentItem]

class SplitConfig(BaseModel):
    delete_temp_file: bool
    nltk_data: str | None
    max_file_size_in_mb: float
    supported_file_types: List[str]
    chunk_size: int
    chunk_overlap: int

@router.get(
    "/split/config",
    response_model=SplitConfig,
    description="Get the current configurations for the split endpoint"
)
async def get_config():
    return SplitConfig(
        delete_temp_file=delete_temp_file,
        nltk_data=nltk_data,
        max_file_size_in_mb=max_file_size_in_mb,
        supported_file_types=supported_file_types.split(",") if supported_file_types else [],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )    


@router.post(
    "/split",
    response_model=Document,
)
async def load_split(
    file: UploadFile = File(...),
    q_chunk_size: int = Query(
        chunk_size, description='Maximum size of chunks in characters to return'),
    q_chunk_overlap: int = Query(
        chunk_overlap, description='Overlap in characters between chunks')
):
    file_chunk_size = 1024 * 1024  # 1 MB

    with tempfile.NamedTemporaryFile(
            mode='wb',
            buffering=file_chunk_size,
            delete=delete_temp_file) as temp:
        while True:
            chunk = await file.read(file_chunk_size)
            if not chunk:
                break
            temp.write(chunk)
        temp.flush()
        docs, mime_type = load(temp)
        docs_len = len(docs)
        print("mime_type", mime_type, ', docs_len', docs_len)

        if docs_len > 1:
            items = []
            content = ''
            for doc in docs:
                items.extend(split(doc, q_chunk_size, q_chunk_overlap))
                content += doc.page_content
            document = Document(
                # None when the source doc is text/plain
                content=content,
                mime_type=mime_type,
                items=items,
            )
            return document
        else:     
            doc = docs[0]
            items = split(doc, q_chunk_size, q_chunk_overlap)
            document = Document(
                # None when the source doc is text/plain
                content=doc.page_content if mime_type != "text/plain" else None,
                mime_type=mime_type,
                items=items,
            )
            return document


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        create_app(), host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000))
    )
elif os.getenv("RUNTIME") == "aws-lambda":
    from mangum import Mangum
    handler = Mangum(create_app())
