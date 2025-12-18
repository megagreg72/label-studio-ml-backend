FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt .
COPY requirements-base.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-base.txt

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
