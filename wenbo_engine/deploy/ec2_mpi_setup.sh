#!/usr/bin/env bash
# Bootstrap an Ubuntu 22.04 i3en instance for MPI wenbo_engine.
# Formats local NVMe, installs OpenMPI + mpi4py + Python deps.
# Run on EVERY node.
#
# Usage:
#   ./ec2_mpi_setup.sh

set -euo pipefail

NVME_MOUNT="/mnt/nvme"

echo "=== [1/4] System packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3 python3-pip python3-venv python3-dev \
    openmpi-bin libopenmpi-dev \
    nvme-cli \
    git htop tmux

echo "=== [2/4] Format + mount NVMe ==="
# i3en.3xlarge has 1x 7.5 TB NVMe drive (plus root EBS).
NVME_DEVS=()
for dev in /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1; do
    if [ -b "$dev" ] && ! lsblk "$dev" 2>/dev/null | grep -q part; then
        NVME_DEVS+=("$dev")
    fi
done

echo "Found ${#NVME_DEVS[@]} unmounted NVMe drive(s): ${NVME_DEVS[*]}"

if [ ${#NVME_DEVS[@]} -ge 1 ]; then
    sudo mkfs.ext4 -F "${NVME_DEVS[0]}"
    sudo mkdir -p "$NVME_MOUNT"
    sudo mount "${NVME_DEVS[0]}" "$NVME_MOUNT"
    sudo chmod 777 "$NVME_MOUNT"
    echo "NVMe ${NVME_DEVS[0]} mounted at $NVME_MOUNT"
else
    echo "No unmounted NVMe found. Falling back to /tmp"
    NVME_MOUNT="/tmp"
fi

echo "=== [3/4] Python deps ==="
sudo pip3 install numpy mpi4py pytest numba 2>/dev/null || \
    sudo pip3 install --break-system-packages numpy mpi4py pytest numba

echo "=== [4/4] Work directory + disk check ==="
WORK_DIR="${NVME_MOUNT}/wenbo_data"
mkdir -p "$WORK_DIR"

echo "Disk write speed:"
dd if=/dev/zero of="$WORK_DIR/_test" bs=1M count=1024 conv=fdatasync 2>&1 | tail -1
rm -f "$WORK_DIR/_test"

# Verify MPI works
echo ""
echo "MPI version:"
mpirun --version | head -1
python3 -c "from mpi4py import MPI; print(f'mpi4py OK, rank {MPI.COMM_WORLD.Get_rank()}')"

echo ""
echo "========================================="
echo "  Done.  NVMe=$NVME_MOUNT"
echo "  Work dir=$WORK_DIR"
echo "========================================="
echo ""
echo "Next: run mpi_cluster.sh on the master node to set up SSH + hostfile"
