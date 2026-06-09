#!/usr/bin/env bash
# spark_cluster.sh — Start/stop Spark master and workers
#
# Usage:
#   ./spark_cluster.sh start-master
#   ./spark_cluster.sh start-worker <master-ip>
#   ./spark_cluster.sh stop
#   ./spark_cluster.sh status
#
# Security group must allow within-SG:
#   - TCP 7077 (Spark master)
#   - TCP 8080 (Spark web UI)
#   - TCP 2049 (NFS)
#   - TCP 111  (NFS portmapper)
#   - All TCP ephemeral ports (Spark shuffle/executor)

set -euo pipefail

SPARK_HOME="${SPARK_HOME:-/opt/spark}"
# Tuned for i3en.xlarge (4 vCPU, 32 GB RAM, 1× 2.5 TB NVMe)
# Master runs NO executor — only Spark master + driver + NFS.
# Workers get most resources for compute.
WORKER_CORES=3           # leave 1 core for OS
WORKER_MEMORY="24g"      # leave 8 GB for OS + buffers

if [ $# -lt 1 ]; then
    echo "Usage: $0 {start-master|start-worker <master-ip>|stop|status}"
    exit 1
fi
ACTION="$1"

configure_spark_defaults() {
    local master_url="$1"
    cat > "${SPARK_HOME}/conf/spark-defaults.conf" <<EOF
spark.master                     ${master_url}
spark.driver.memory              4g
spark.executor.memory            24g
spark.executor.cores             3
spark.default.parallelism        6
spark.pyspark.python             python3

# Serialization
spark.serializer                 org.apache.spark.serializer.KryoSerializer

# Network timeouts (important for large chunk I/O)
spark.network.timeout            600s
spark.executor.heartbeatInterval 60s

# Local dirs — use second NVMe for shuffle/spill, first NVMe for RDD disk
# Both paths used: Spark stripes across them for parallelism
spark.local.dir                  /mnt/nvme2/spark_tmp,/mnt/nvme/spark_tmp

# Data locality — wait for PROCESS_LOCAL scheduling
spark.locality.wait              10s
spark.locality.wait.process      10s

# Task scheduling
spark.task.maxFailures           4
spark.speculation                false

# Logging
spark.eventLog.enabled           false
EOF
    echo "Wrote spark-defaults.conf for master=${master_url}"
}

case "$ACTION" in
    start-master)
        MASTER_IP=$(hostname -I | awk '{print $1}')
        MASTER_URL="spark://${MASTER_IP}:7077"

        configure_spark_defaults "$MASTER_URL"

        # Export /mnt/nvme/wenbo_data via NFS so all workers share it
        WORK_DIR="/mnt/nvme/wenbo_data"
        mkdir -p "$WORK_DIR"
        # Allow all private IPs in the VPC subnet
        echo "${WORK_DIR} 172.31.0.0/16(rw,sync,no_subtree_check,no_root_squash)" \
            | sudo tee /etc/exports > /dev/null
        sudo exportfs -ra
        sudo systemctl restart nfs-kernel-server
        echo "NFS: exporting ${WORK_DIR} to 172.31.0.0/16"

        "${SPARK_HOME}/sbin/start-master.sh" \
            --host "$MASTER_IP" \
            --port 7077 \
            --webui-port 8080

        echo ""
        echo "Spark master started at: $MASTER_URL"
        echo "Web UI: http://${MASTER_IP}:8080"
        echo ""
        echo "NOTE: Master runs NO executor — dedicated to driver + NFS."
        echo "On each WORKER node (not this one), run:"
        echo "  ./spark_cluster.sh start-worker ${MASTER_IP}"
        ;;

    start-worker)
        MASTER_IP="${2:?Usage: $0 start-worker <master-ip>}"
        MASTER_URL="spark://${MASTER_IP}:7077"

        configure_spark_defaults "$MASTER_URL"

        # Mount master's NFS share so all nodes see the same data dir
        WORK_DIR="/mnt/nvme/wenbo_data"
        sudo mkdir -p "$WORK_DIR"
        if ! mount | grep -q "$WORK_DIR"; then
            sudo mount -t nfs "${MASTER_IP}:${WORK_DIR}" "$WORK_DIR"
            echo "NFS: mounted ${MASTER_IP}:${WORK_DIR}"
        fi

        # Ensure spark tmp dirs exist on both NVMe drives
        mkdir -p /mnt/nvme/spark_tmp /mnt/nvme2/spark_tmp 2>/dev/null || true

        "${SPARK_HOME}/sbin/start-worker.sh" \
            "$MASTER_URL" \
            --cores "$WORKER_CORES" \
            --memory "$WORKER_MEMORY"

        echo "Worker started, connected to $MASTER_URL"
        echo "  Cores: $WORKER_CORES"
        echo "  Memory: $WORKER_MEMORY"
        ;;

    stop)
        "${SPARK_HOME}/sbin/stop-all.sh" 2>/dev/null || true
        "${SPARK_HOME}/sbin/stop-worker.sh" 2>/dev/null || true
        "${SPARK_HOME}/sbin/stop-master.sh" 2>/dev/null || true
        echo "Spark processes stopped."
        ;;

    status)
        echo "=== Spark processes ==="
        ps aux | grep -E '[s]park' || echo "(none running)"
        echo ""
        echo "=== Java processes ==="
        jps 2>/dev/null || echo "(jps not available)"
        ;;

    *)
        echo "Usage: $0 {start-master|start-worker <master-ip>|stop|status}"
        exit 1
        ;;
esac
