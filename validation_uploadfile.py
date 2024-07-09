from typing import List
from enum import Enum

from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette import status

class FileTypeName(str, Enum):
    JPEG = "image/jpeg"
    JPG = "image/jpeg"
    PNG = "image/png"
    GIF = "image/gif"
    WEBP = "image/webp"
    PDF = "application/pdf"
    ZIP = "application/zip"
    TXT = "text/plain"

class ValidateUploadFileMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app: ASGIApp,
        app_paths: List[str] = None,
        max_size: int = 16 * 1024 * 1024,  # 16MB in bytes
        file_types: List[FileTypeName] = None
    ) -> None:
        super().__init__(app)
        self.app_paths = app_paths or []
        self.max_size = max_size
        self.file_types = file_types or []

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method not in {"POST", "PUT"}:
            return await call_next(request)
        
        if request.url.path in self.app_paths:
            try:
                request._body = await request.body()
                form = await request.form()
          
                if not form:
                    # Handle case where there are no files
                    return PlainTextResponse("No files provided", status_code=status.HTTP_400_BAD_REQUEST)

                file = next(iter(form))
                content_type = form[file].content_type
                print('content_type', content_type)
                if self.file_types and content_type not in self.file_types:
                    return PlainTextResponse("Unsupported Media Type", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
                
                content_length = int(request.headers.get("content-length", 0))
                if content_length == 0:
                    return PlainTextResponse("Length Required", status_code=status.HTTP_411_LENGTH_REQUIRED)

                if content_length > self.max_size:
                    return PlainTextResponse("Request Entity Too Large", status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

            except Exception as e:
                return PlainTextResponse(str(e), status_code=status.HTTP_400_BAD_REQUEST)

        return await call_next(request)
