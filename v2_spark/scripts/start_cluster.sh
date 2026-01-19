#!/bin/bash
# Start a real Spark cluster with multiple workers for distribution testing

set -e

echo "=============================================="
echo "Starting Spark Cluster for Distribution Testing"
echo "=============================================="

cd "$(dirname "$0")/.."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

echo ""
echo "Step 1: Starting Spark cluster (1 master + 3 workers)..."
docker-compose -f docker-compose-cluster.yml up -d

echo ""
echo "Step 2: Waiting for cluster to initialize..."
sleep 15

echo ""
echo "Step 3: Checking cluster status..."
echo ""
echo "Spark Master UI: http://localhost:8080"
echo "Worker 1 UI: http://localhost:8081"
echo "Worker 2 UI: http://localhost:8082"
echo "Worker 3 UI: http://localhost:8083"
echo ""

# Check if master is up
if curl -s http://localhost:8080 > /dev/null 2>&1; then
    echo "✓ Spark Master is running"
    
    # Try to get worker count
    WORKERS=$(curl -s http://localhost:8080/json/ 2>/dev/null | \
        python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('workers', [])))" 2>/dev/null || echo "?")
    
    echo "  Workers registered: $WORKERS"
else
    echo "⚠️  Spark Master not responding yet (may need more time)"
fi

echo ""
echo "Step 4: To run distribution test:"
echo "  docker-compose -f docker-compose-cluster.yml run --rm quantum-simulator \\"
echo "    python3 scripts/verify_real_distribution.py"
echo ""
echo "Step 5: To stop cluster:"
echo "  docker-compose -f docker-compose-cluster.yml down"
echo ""
