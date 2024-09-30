FROM python:3.11-slim-bookworm

RUN pip install mlflow boto3 psycopg2-binary

EXPOSE ${MLFLOW_PORT}

ENTRYPOINT ["mlflow"]
CMD ["server", "--default-artifact-root", "s3://mlflow/", "--host", "0.0.0.0"]