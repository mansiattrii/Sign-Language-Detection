from pathlib import Path

import pytest
from fastapi.testclient import TestClient

DATASET = Path("data/processed/include.npz")
CACHE = Path("data/processed/landmarks/include")


def pytest_runtest_setup(item):
    if item.get_closest_marker("needs_data") and not (DATASET.exists() and CACHE.exists()):
        pytest.skip("needs the extracted INCLUDE data")


@pytest.fixture(scope="session")
def client():
    from api.app import app

    with TestClient(app) as client:
        yield client
