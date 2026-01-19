# Quantum Simulator with Spark support
FROM eclipse-temurin:17-jre-jammy

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        curl \
        procps \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install Python dependencies
# Pin PySpark to 3.5.x for stability
RUN pip3 install --no-cache-dir \
    'pyspark>=3.5,<3.6' \
    'numpy>=1.21,<2' \
    'duckdb>=0.9' \
    'pandas>=1.3' \
    'pytest>=7.0' \
    'opt_einsum==3.3.0'

# Set SPARK_HOME to PySpark's bundled Spark
ENV SPARK_HOME=/usr/local/lib/python3.10/dist-packages/pyspark
ENV PATH="$SPARK_HOME/bin:$PATH"
ENV PYSPARK_PYTHON=python3

# Copy application code
COPY . /app

ENV PYTHONPATH=/app

# Default command
CMD ["python3", "-m", "pytest", "tests/", "-v"]
