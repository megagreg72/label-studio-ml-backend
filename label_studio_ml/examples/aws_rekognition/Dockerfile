FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install base requirements
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model code
COPY model.py .
COPY _wsgi.py .

# Expose port
EXPOSE 9090

# Set environment variables
ENV FLASK_APP=_wsgi.py
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "_wsgi.py"]
