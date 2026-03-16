FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt ./server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy entire project (frontend + server)
COPY . .

# Cloud Run listens on 8080 by default
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
