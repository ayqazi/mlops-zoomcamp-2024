FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install pipenv

COPY Pipfile Pipfile.lock ./
RUN pipenv sync

COPY predict.py ./
