# Base image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Expose HF Space port
EXPOSE 7860

# Prevent analytics errors
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run your Streamlit app
CMD ["streamlit", "run", "UI/app2.py", "--server.port=7860", "--server.address=0.0.0.0"]
