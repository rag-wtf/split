from typing import List, Optional
from enum import Enum

from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
import logging # For logging
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette import status
from starlette.datastructures import UploadFile

class FileTypeName(str, Enum):
    """
    An enumeration of commonly supported MIME types.

    This enum provides a standardized way to refer to various file types
    by their MIME type strings. It's used by ValidateUploadFileMiddleware
    to define which file types are permissible.
    """
    JPEG = "image/jpeg"
    JPG = "image/jpg"
    PNG = "image/png"
    GIF = "image/gif"
    WEBP = "image/webp"
    PDF = "application/pdf"
    ZIP = "application/zip"
    TXT = "text/plain"
    HTML = "text/html"
    MD = "text/markdown"
    PPT = "application/vnd.ms-powerpoint"
    ODP = "application/vnd.openxmlformats-officedocument.presentationml.presentation" 
    DOC = "application/msword"
    ODT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    EPUB = "application/epub+zip"
    EML = "message/rfc822"
    GZIP = "application/gzip"
    
logger = logging.getLogger(__name__)

class ValidateUploadFileMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for validating uploaded files based on size and MIME type.

    This middleware intercepts incoming HTTP requests to specified application paths.
    For POST or PUT requests, it attempts to parse the form data to access uploaded
    files. It then checks each file against configured maximum size limits and
    allowed MIME types. If validation fails, it returns an appropriate HTTP error
    response (e.g., 400, 411, 413, 415).

    Attributes:
        app_paths (List[str]): A list of URL paths where this middleware should
            apply its validation logic.
        max_size (int): The maximum allowed file size in bytes.
        file_types (List[FileTypeName]): A list of allowed MIME types (as strings
            or using the FileTypeName enum).
    """
    def __init__(
        self,
        app: ASGIApp,
        app_paths: Optional[List[str]] = None,
        max_size: int = 16 * 1024 * 1024,  # 16MB in bytes
        file_types: Optional[List[FileTypeName]] = None
    ) -> None:
        super().__init__(app)
        self.app_paths = app_paths or []
        self.max_size = max_size
        self.file_types = file_types or []

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        Core middleware logic to validate uploaded files in a request.

        This method is called for each request. It checks if the request method
        is POST/PUT and if the URL path matches one of the configured app_paths.
        If so, it attempts to:
        1. Read the request body and parse it as a form.
        2. Check the Content-Length header for size constraints.
        3. Check the Content-Type of the uploaded file(s) against allowed types.

        Args:
            request (Request): The incoming Starlette/FastAPI request object.
            call_next (RequestResponseEndpoint): A function that will call the
                next middleware in the chain or the actual endpoint handler.

        Returns:
            Response: An HTTP error response if validation fails (e.g.,
                PlainTextResponse with status codes 400, 411, 413, 415),
                or the response from `call_next(request)` if validation
                is successful or not applicable.
        """
        if request.method not in {"POST", "PUT"}:
            return await call_next(request)
        
        if request.url.path in self.app_paths:
            try:
                request._body = await request.body() # Consume body once
                form = await request.form()
          
                if not form:
                    # Handle case where there are no files
                    return PlainTextResponse("No files provided", status_code=status.HTTP_400_BAD_REQUEST)

                all_files_valid = True
                # Iterate through all form items to find UploadFile instances
                for key in form:
                    for item_in_list in form.getlist(key):
                        if isinstance(item_in_list, UploadFile): # It's an UploadFile object
                            upload_file_obj = item_in_list 
                            content_type = upload_file_obj.content_type
                            logger.debug(f"Validating file: {upload_file_obj.filename}, content_type: {content_type}")
                            if self.file_types and content_type not in self.file_types:
                                all_files_valid = False
                                break # Stop checking files for this key
                    if not all_files_valid:
                        break # Stop checking other keys
            
                if not all_files_valid:
                    return PlainTextResponse("Unsupported Media Type", status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
                
                # Content length validation (remains the same, applied to overall request size)
                content_length = int(request.headers.get("content-length", 0))
                if content_length == 0:
                    return PlainTextResponse("Length Required", status_code=status.HTTP_411_LENGTH_REQUIRED)

                if content_length > self.max_size:
                    return PlainTextResponse("Request Entity Too Large", status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)

            except Exception as e:
                return PlainTextResponse(str(e), status_code=status.HTTP_400_BAD_REQUEST)

        return await call_next(request)
