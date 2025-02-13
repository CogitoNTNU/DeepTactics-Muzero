from fastapi.testclient import TestClient
from src.app import app


client = TestClient(app)

def test_prediction_without_arguments():
    response = client.post("/predict")
    assert response.status_code == 422



