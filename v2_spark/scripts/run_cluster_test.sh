#!/bin/bash
# Run quantum simulator on a real Spark cluster using Docker

set -e

echo "=============================================="
echo "Spark Cluster Test for Quantum Simulator"
echo "=============================================="

cd "$(dirname "$0")/.."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

echo ""
echo "Step 1: Starting Spark cluster..."
docker-compose up -d spark-master spark-worker-1 spark-worker-2

echo ""
echo "Waiting for cluster to be ready..."
sleep 10

echo ""
echo "Step 2: Checking cluster status..."
docker exec spark-master curl -s http://localhost:8080/json/ | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(f\"  Master: {data.get('url', 'unknown')}\")
print(f\"  Workers: {len(data.get('workers', []))}\")
print(f\"  Cores: {data.get('cores', 0)}\")
print(f\"  Memory: {data.get('memory', 0) / 1024:.1f} GB\")
" 2>/dev/null || echo "  (waiting for API...)"

echo ""
echo "Step 3: Building simulator container..."
docker-compose build quantum-simulator

echo ""
echo "Step 4: Running distribution verification test..."
docker-compose run --rm quantum-simulator python3 scripts/verify_distribution.py

echo ""
echo "Step 5: Running test suite on cluster..."
docker-compose run --rm quantum-simulator python3 -m pytest tests/ -v --tb=short -x

echo ""
echo "Step 6: Running scalability test on cluster..."
docker-compose run --rm quantum-simulator python3 scripts/scalability_test.py

echo ""
echo "=============================================="
echo "Cluster test complete!"
echo "=============================================="
echo ""
echo "Spark Master UI: http://localhost:8080"
echo ""
echo "To stop the cluster: docker-compose down"
