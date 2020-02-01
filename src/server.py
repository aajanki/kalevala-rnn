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

    verses = sampler.sample_verses(0.2, 300)
    return templates.TemplateResponse('main.html', {'request': request, 'verses': verses})
