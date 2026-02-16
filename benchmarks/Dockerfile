FROM python:3.12-slim

WORKDIR /app

# Install build deps
RUN pip install --no-cache-dir setuptools wheel "setuptools_scm[toml]"

# Install comparison frameworks first (these change rarely, so cache this layer)
RUN pip install --no-cache-dir langgraph llama-index-core prefect dagster psutil

# Copy project and install puffinflow (this layer invalidates when code changes)
COPY . .
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0
RUN pip install --no-cache-dir -e .

# Suppress noisy Prefect/Dagster logs
ENV PREFECT_LOGGING_LEVEL=WARNING
ENV PREFECT_SILENCE_API_URL_MISCONFIGURATION=true
ENV PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS=60
ENV DAGSTER_IS_DEV_CLI=false

CMD ["python", "benchmarks/benchmark.py", "--json", "--runs", "20", "-n", "5000"]
