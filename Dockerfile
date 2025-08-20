# Build a tiny runnable container for the CLI
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy project for install
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Default entrypoint runs the CLI (expects CSVs in CWD or via flags)
ENTRYPOINT ["haz-subst"]
# Example:
# docker run --rm -v "$PWD":/work -w /work ghcr.io/tomermaas/hazarot:latest #   --book book_qusetions.csv --containers-questions containers_questions.csv
