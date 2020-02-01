import os
from fastapi import FastAPI
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from .sample import TextSampler, take

app = FastAPI(title='Kalevalakone')
templates = Jinja2Templates(directory='templates')
sampler = None


@app.on_event("startup")
def startup_event():
    global sampler
    sampler = TextSampler(os.getenv('MODEL_PATH'))


@app.get("/")
async def root(request: Request):
    global sampler

    characters = sampler.character_generator(0.2)
    verses = ''.join(take(300, characters))

    return templates.TemplateResponse('main.html', {'request': request, 'verses': verses})
