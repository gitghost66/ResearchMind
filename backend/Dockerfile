FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy packages
COPY agents /app/agents
COPY backend /app/backend

# Install agents package then backend
RUN pip install --no-cache-dir -e /app/agents
RUN pip install --no-cache-dir -e /app/backend

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
