import os
from ingest import create_app
from dotenv import load_dotenv
load_dotenv()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        create_app(), host=os.getenv("HOST", "localhost"), port=int(os.getenv("PORT", 8000))
    )
