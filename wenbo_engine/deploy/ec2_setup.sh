#!/usr/bin/env bash
# Bootstrap an Ubuntu 22.04 i3en instance for wenbo_engine.
# Formats the local NVMe, installs Java + Spark + Python deps.
# Run on EVERY node (master + workers).
#
# Usage:
#   ./ec2_setup.sh

set -euo pipefail

SPARK_VERSION="3.5.8"
SPARK_HADOOP="spark-${SPARK_VERSION}-bin-hadoop3"
SPARK_HOME="/opt/spark"
NVME_MOUNT="/mnt/nvme"

echo "=== [1/5] System packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv \
    default-jre-headless \
    nvme-cli \
    nfs-kernel-server nfs-common \
    git htop tmux

echo "=== [2/5] Format + mount NVMe ==="
# i3en instances have instance store NVMe at /dev/nvme1n1 (or similar)
NVME_DEV=""
for dev in /dev/nvme1n1 /dev/nvme2n1 /dev/nvme0n1; do
    # skip the root EBS volume
    if [ -b "$dev" ] && ! mount | grep -q "$dev"; then
        NVME_DEV="$dev"
        break
    fi
done

if [ -z "$NVME_DEV" ]; then
    echo "No unmounted NVMe found. Available block devices:"
    lsblk
    echo "Falling back to /tmp"
    NVME_MOUNT="/tmp"
else
    sudo mkfs.ext4 -F "$NVME_DEV"
    sudo mkdir -p "$NVME_MOUNT"
    sudo mount "$NVME_DEV" "$NVME_MOUNT"
    sudo chmod 777 "$NVME_MOUNT"
    echo "NVMe $NVME_DEV mounted at $NVME_MOUNT"
fi

echo "=== [3/5] Install Spark ==="
if [ ! -d "$SPARK_HOME" ]; then
    cd /tmp
    wget -q "https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}/${SPARK_HADOOP}.tgz"
    sudo tar xzf "${SPARK_HADOOP}.tgz" -C /opt
    sudo ln -sf "/opt/${SPARK_HADOOP}" "$SPARK_HOME"
    sudo chown -R ubuntu:ubuntu "/opt/${SPARK_HADOOP}"
    rm -f "${SPARK_HADOOP}.tgz"
fi

cat <<'EOF' | sudo tee /etc/profile.d/spark.sh
export SPARK_HOME=/opt/spark
export PATH=$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH
export PYSPARK_PYTHON=python3
EOF
source /etc/profile.d/spark.sh

echo "=== [4/5] Python deps ==="
sudo pip3 install numpy pyspark==3.5.8 pytest 2>/dev/null || \
    sudo pip3 install --break-system-packages numpy pyspark==3.5.8 pytest

echo "=== [5/5] Work directory ==="
WORK_DIR="${NVME_MOUNT}/wenbo_data"
mkdir -p "$WORK_DIR"
echo "Work dir: $WORK_DIR"

# quick disk speed check
echo "Disk write speed:"
dd if=/dev/zero of="$WORK_DIR/_test" bs=1M count=512 conv=fdatasync 2>&1 | tail -1
rm -f "$WORK_DIR/_test"

echo ""
echo "========================================="
echo "  Done."
echo "  SPARK_HOME=$SPARK_HOME"
echo "  NVMe=$NVME_MOUNT"
echo "  Work dir=$WORK_DIR"
echo "========================================="
echo ""
echo "Next:"
echo "  0. Security group: open ports 7077, 8080, 2049, 111 + all TCP within SG"
echo "  1. Upload code:  scp -r wenbo_engine/ ubuntu@<this-ip>:~/wenbo_engine/"
echo "  2. Master:       bash wenbo_engine/deploy/spark_cluster.sh start-master"
echo "  3. Workers:      bash wenbo_engine/deploy/spark_cluster.sh start-worker <master-private-ip>"
echo "  4. Create zip:   python3 wenbo_engine/deploy/make_zip.py"
echo "  5. Smoke test:   spark-submit --py-files wenbo_engine.zip wenbo_engine/deploy/smoke_test.py"
