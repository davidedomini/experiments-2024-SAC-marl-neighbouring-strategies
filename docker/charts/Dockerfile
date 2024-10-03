FROM python:3.10
RUN python -m pip install poetry
RUN mkdir /experiment
WORKDIR /experiment
COPY . .
ENV PYTHONWARNINGS="ignore::DeprecationWarning"
RUN poetry install
CMD poetry run python src/plotter.py