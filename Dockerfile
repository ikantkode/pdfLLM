FROM python:3.11-slim

# Install system dependencies (merged from exaOCR and pdfLLM)
RUN apt-get update && apt-get install -y \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    ghostscript \
    libreoffice \
    fonts-dejavu \
    unpaper \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user for Celery and other services
RUN useradd -m -u 1000 appuser

# Create data and logs directories with correct permissions as root
RUN mkdir -p /app/data/logs && chown -R appuser:appuser /app/data && chmod -R u+rwX /app/data

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model (from pdfLLM)
RUN python -m spacy download en_core_web_sm

# Copy the entire project
COPY . .

# Set ownership for non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]