FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    tesseract-ocr \
    libtesseract-dev \
    ghostscript \
    libmupdf-dev \
    mupdf \
    libjpeg-dev \
    zlib1g-dev \
    libreoffice \
    poppler-utils \
    libexif-dev \
    exiftool \
    git \
    python3-tk \
    pandoc \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/microsoft/markitdown.git /markitdown && \
    cd /markitdown && \
    pip install -e 'packages/markitdown[all]'
RUN markitdown --version

RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/lib /usr/lib
COPY --from=builder /usr/share/tesseract-ocr /usr/share/tesseract-ocr

RUN apt-get update && apt-get install -y \
    libpq5 \
    tesseract-ocr \
    ghostscript \
    mupdf \
    libreoffice \
    poppler-utils \
    libexif-dev \
    exiftool \
    python3-tk \
    pandoc \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY app/main.py main.py
RUN ls -la /app && \
    test -f /app/main.py || (echo "ERROR: main.py not found after COPY" && exit 1) && \
    cat /app/main.py && \
    chmod 644 /app/main.py && \
    python -c "import sys; sys.path.append('/app'); import main" && \
    ls -la /app/app 2>/dev/null || echo "No /app/app directory"

CMD ["/bin/sh", "-c", "ls -la /app && python -c 'import sys; sys.path.append(\"/app\"); import main' && uvicorn main:app --host 0.0.0.0 --port 8000"]