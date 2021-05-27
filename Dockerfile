FROM 3.8-slim

RUN mkdir /app

WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY src /app/

COPY models /app/

COPY config.ini /app/

CMD ["uvicorn app:app --port 8000 --reload", "/app/main.py"]