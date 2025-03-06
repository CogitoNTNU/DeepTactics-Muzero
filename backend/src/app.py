from typing import Union

from fastapi import FastAPI

app = FastAPI()

# TODO add CORS middleware
# Use this as some basic security feature.
# Will make it harder for attackers from other domains to access the backend API.
"""
origins = [
    "http://localhost",
    "http://localhost:8080",
    # frontend container.
    # cogito websites
]
"""

# Websockets ?? Interesting

@app.post("/predict")
async def predict(game_type: str, game_state: Union[str, dict]) -> dict:
    return {"game_type": game_type, "game_state": game_state}
