FROM python:3.9

WORKDIR /app

# Add SSL and system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install torch directly
RUN pip install --upgrade pip && \
    pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html \
    --trusted-host download.pytorch.org --timeout 1200 --retries 5

# Copy requirements and install rest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY explainers/ ./explainers/
COPY utils/ ./utils/
COPY run_demo.py .
COPY xai_app.py .
COPY xai_pipeline.py .

CMD ["uvicorn", "xai_app:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000
