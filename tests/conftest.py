import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    from api.app import app

    with TestClient(app) as client:
        yield client
