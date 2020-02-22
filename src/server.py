import os
import re
from fastapi import FastAPI, Form
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from typing import List, Optional
from .sample import load_sampler, KeywordSampler


app = FastAPI(title='Kalevala-kone',
              openapi_url=None,
              docs_url=None,
              redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')
sampler = None


@app.on_event("startup")
def startup_event():
    global sampler
    sampler = load_sampler(os.getenv('KALEVALA_MODEL_PATH'))


@app.get("/")
async def main(request: Request):
    verses = generate_verses([])
    return templates.TemplateResponse('main.html', {'request': request, 'verses': verses})


@app.get("/api/verses")
async def verses_api(keywords: str = None):
    return generate_verses(split_keywords(keywords))


def generate_verses(keywords: List[str]) -> List[str]:
    sampler.reset_states()
    temperature = float(os.getenv('KALEVALA_TEMPERATURE', 0.2))
    keyword_sampler = KeywordSampler(keywords)
    return sampler.sample_verses(temperature, 120, keywords=keyword_sampler)


def split_keywords(keyword_string: Optional[str]) -> List[str]:
    if not keyword_string:
        return []
    else:
        keywords = re.split(r'[^a-zåäöA-ZÅÄÖ]+', keyword_string)
        keywords = (x for x in keywords if len(x) > 1 and len(x) < 15)

        # The RNN seems to generate slightly better verses if we start
        # lines with capital letters like in the source material.
        return [x.capitalize() for x in keywords]
