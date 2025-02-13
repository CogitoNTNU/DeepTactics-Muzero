from typing import Union

from fastapi import FastAPI

app = FastAPI()

# Websockets ?? Interesting

@app.post("/predict")
async def predict(game_type: str, game_state: Union[str, dict]) -> dict:
    return {"game_type": game_type, "game_state": game_state}
