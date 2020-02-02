import os
import re
from fastapi import FastAPI, Form
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from typing import List, Optional
from .sample import load_sampler, KeywordSampler


app = FastAPI(title='Kalevala-kone')
templates = Jinja2Templates(directory='templates')
sampler = None


@app.on_event("startup")
def startup_event():
    global sampler
    sampler = load_sampler(os.getenv('MODEL_PATH'))


@app.get("/")
async def main(request: Request):
    return await verses_page(request)


@app.post("/")
async def verses_with_keywords(request: Request, keywords: str = Form(...)):
    return await verses_page(request, keywords)


async def verses_page(request: Request, keywords: Optional[str]=None):
    global sampler

    sampler.reset_states()
    keyword_sampler = KeywordSampler(split_keywords(keywords))
    verses = sampler.sample_verses(0.2, 300, keywords=keyword_sampler)
    return templates.TemplateResponse('main.html', {'request': request, 'verses': verses})


def split_keywords(keyword_string: Optional[str]) -> List[str]:
    if not keyword_string:
        return []
    else:
        keywords = re.split(r'[^a-zåäöA-ZÅÄÖ]+', keyword_string)
        keywords = (x for x in keywords if len(x) > 1 and len(x) < 15)

        # RNN seems to generate slightly better verses if we start
        # lines with capital letters like in the source material.
        return [x.capitalize() for x in keywords]
