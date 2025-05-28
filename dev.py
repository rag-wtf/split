from split import create_app, settings # Import settings
from dotenv import load_dotenv
load_dotenv() # pydantic-settings will also try to load .env, but this ensures it's loaded early

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        create_app(), host=settings.host, port=settings.port
    )
