# EgoVault — Docker image
#
# Runs the Streamlit browser UI on port 8501.
# llama-server must be reachable from inside the container — either run it on
# the host and pass --add-host=host.docker.internal:host-gateway, or spin it
# up as a separate container and use docker-compose (see docker-compose.yml).
#
# Build:
#   docker build -t egovault .
#
# Run (connects to llama-server on the host):
#   docker run -p 8501:8501 \
#     -v ./data:/app/data \
#     -v ./inbox:/app/inbox \
#     --add-host=host.docker.internal:host-gateway \
#     egovault

FROM python:3.12-slim

# System deps for PDF/OCR and lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[local,gmail,websearch]"

# Copy source
COPY egovault/ egovault/
COPY ego.sh .
COPY data/agent.md data/agent.md 2>/dev/null || true

# Runtime directories (overridden by volume mounts)
RUN mkdir -p data inbox output

EXPOSE 8501

# Streamlit browser UI — no WAN tunnel inside Docker (use --no-wan)
CMD ["egovault", "web", "--no-wan", "--host", "0.0.0.0", "--port", "8501"]
