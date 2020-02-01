import os
from fastapi import FastAPI
from .sample import load_inference_model, character_generator, take

app = FastAPI()
model, char2idx, idx2char = load_inference_model(os.getenv('MODEL_PATH'))


@app.get("/")
async def root():
    preseed = 'Mieleni minun tekevi'
    characters = character_generator(
        model, 0.2, idx2char, char2idx, preseed)
    text = preseed + ''.join(take(200, characters))
    return {'content': text}
