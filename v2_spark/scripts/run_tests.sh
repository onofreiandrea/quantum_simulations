#!/bin/bash
# Run all tests in Docker container

set -e

echo "========================================="
echo "Building Docker image..."
echo "========================================="
docker build -t quantum-spark-test .

echo ""
echo "========================================="
echo "Running tests..."
echo "========================================="
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    quantum-spark-test \
    pytest tests/ -v --tb=short

echo ""
echo "========================================="
echo "All tests completed!"
echo "========================================="
