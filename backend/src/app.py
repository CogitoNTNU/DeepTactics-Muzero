from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [ # Allow frontend to access the backend
    "http://localhost:9135",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Websockets ?? Interesting

@app.post("/predict")
async def predict(game_type: str, game_state: Union[str, dict]) -> dict:
    return {"game_type": game_type, "game_state": game_state}
