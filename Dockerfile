# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - build-essential: needed for some python packages that compile C extensions
# - libxml2/libxslt: needed by python-pptx (via lxml)
# - fonts-dejavu-core: gives reportlab sane default fonts for poster PDF export
# - curl: handy for container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create runtime directories used for uploads and generated/exported assets
# (uploads, generated images/videos, exported pptx/pdf files)
RUN mkdir -p static/uploads static/generated static/exports

# Expose default port
EXPOSE 8000

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Basic healthcheck against the app's own /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run as a non-root user for better container security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Start the application using uvicorn
CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]