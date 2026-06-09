# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data
COPY .env.example .env.example

# Install dependencies
RUN uv sync --frozen --no-dev

# Create volume mount points
VOLUME ["/app/data"]

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the local web service by default.
ENTRYPOINT ["uv", "run"]
CMD ["ai-cto-daily-web", "--host", "0.0.0.0", "--port", "8765"]
