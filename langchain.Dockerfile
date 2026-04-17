FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY langchain/requirements.txt /app/langchain/requirements.txt
RUN pip install --no-cache-dir -r /app/langchain/requirements.txt

COPY langchain/app /app/langchain/app

EXPOSE 3000
CMD ["uvicorn", "langchain.app.main:app", "--host", "0.0.0.0", "--port", "3000"]
