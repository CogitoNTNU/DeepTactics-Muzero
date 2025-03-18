from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.post("/predict")
async def predict(game_type: str, game_state: Union[str, dict]) -> dict:
    return {"game_type": game_type, "game_state": game_state}
