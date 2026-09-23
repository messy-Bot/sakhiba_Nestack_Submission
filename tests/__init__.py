from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_query_validation_empty_query():
    response = client.post(
        "/query",
        json={
            "query": "",
            "top_k": 3
        }
    )

    assert response.status_code == 422


def test_query_validation_invalid_top_k():
    response = client.post(
        "/query",
        json={
            "query": "test query",
            "top_k": 0
        }
    )

    assert response.status_code == 422


def test_query_validation_top_k_too_large():
    response = client.post(
        "/query",
        json={
            "query": "test query",
            "top_k": 21
        }
    )

    assert response.status_code == 422
