from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore', case_sensitive=False)

    delete_temp_file: bool = True
    nltk_data: str = "/tmp/nltk_data"  # Default as per original serverless.yml
    max_file_size_in_mb: float = 10.0
    supported_file_types: str = \
        "text/plain,application/pdf,text/html,text/markdown," + \
        "application/vnd.ms-powerpoint,application/vnd.openxmlformats-officedocument.presentationml.presentation," + \
        "application/msword,application/vnd.openxmlformats-officedocument.wordprocessingml.document," + \
        "application/epub+zip,message/rfc822,application/gzip"
    
    chunk_size: int = 500
    chunk_overlap: int = 20
    host: str = "0.0.0.0"
    port: int = 8000
    runtime: Optional[str] = None
    hf_home: str = "/tmp/hf_home" # As per serverless.yml example

settings = Settings()
