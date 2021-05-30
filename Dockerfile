FROM python:3.8-slim

RUN mkdir /app

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY language_models /app/language_models

RUN python3 -m pip install /app/language_models/pl_spacy_model-0.1.0.tar.gz

COPY models /app/models

COPY config.ini /app/

COPY src /app/src

EXPOSE 8100

ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8100", "--reload"]