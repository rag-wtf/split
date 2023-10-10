from ingest import create_app
from dotenv import load_dotenv
load_dotenv()


def test():
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    with open("breathing.gz", "rb") as file:
        response = client.post("/ingest", files={"file": file})
    print(response.content)


if __name__ == "__main__":
    test()
