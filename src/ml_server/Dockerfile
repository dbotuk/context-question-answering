FROM python:3.11-slim-bookworm

WORKDIR /ml_server

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ADD . /ml_server

EXPOSE ${ML_SERVER_PORT}

CMD ["python", "ml_server.py"]