FROM python:3.10

RUN python -m pip install poetry

WORKDIR /experiment

COPY . /experiment/

RUN poetry install

CMD poetry run python src/main.py