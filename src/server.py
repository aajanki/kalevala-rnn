import os
from fastapi import FastAPI
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from .sample import load_inference_model, character_generator, take

app = FastAPI(title='Kalevalakone')
templates = Jinja2Templates(directory='templates')
model, char2idx, idx2char = load_inference_model(os.getenv('MODEL_PATH'))


@app.get("/")
async def root(request: Request):
    preseed = 'Mieleni minun tekevi'
    characters = character_generator(
        model, 0.2, idx2char, char2idx, preseed)
    verses = preseed + ''.join(take(300, characters))

    return templates.TemplateResponse('main.html', {'request': request, 'verses': verses})
