from load_split_embed import create_app
from dotenv import load_dotenv
load_dotenv()


def test():
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    with open("breathing.txt", "rb") as file:
        response = client.post("/load_split_embed", files={"file": file})
    print(response.content)


if __name__ == "__main__":
    test()
