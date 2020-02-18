FROM kennethreitz/pipenv:latest

ENV PIP_NO_CACHE_DIR=off \
    PIPENV_HIDE_EMOJIS=true \
    PIPENV_COLORBLIND=true \
    PIPENV_NOSPIN=true

COPY . /app

CMD uvicorn src.server:app --env-file prod.env --host=0.0.0.0 --port=${PORT:-5000}
