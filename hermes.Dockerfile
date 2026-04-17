FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY hermes/requirements.txt /app/hermes/requirements.txt
RUN pip install --no-cache-dir -r /app/hermes/requirements.txt

COPY hermes/app /app/hermes/app

EXPOSE 3000
CMD ["uvicorn", "hermes.app.main:app", "--host", "0.0.0.0", "--port", "3000"]
