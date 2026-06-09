#!/usr/bin/env bash
# mpi_cluster.sh — Set up passwordless SSH between MPI nodes + create hostfile
#
# Usage (run from master node):
#   ./mpi_cluster.sh setup-ssh  <ip1> <ip2> <ip3> <ip4>
#   ./mpi_cluster.sh create-hostfile <ip1> <ip2> <ip3> <ip4>
#   ./mpi_cluster.sh test <ip1> <ip2> <ip3> <ip4>
#
# Prerequisites:
#   - EC2 SSH key at ~/.ssh/wenbo-mpi-cluster.pem
#   - ec2_mpi_setup.sh already run on all nodes

set -euo pipefail

EC2_KEY="$HOME/.ssh/wenbo-mpi-cluster.pem"
SSH_USER="ubuntu"
MPI_KEY="$HOME/.ssh/mpi_cluster_rsa"
HOSTFILE="$HOME/hostfile"
SLOTS_PER_NODE=1   # 1 MPI rank per node (each rank uses all cores)

if [ $# -lt 1 ]; then
    echo "Usage: $0 {setup-ssh|create-hostfile|test} <ip1> <ip2> ..."
    exit 1
fi
ACTION="$1"
shift
NODES=("$@")

if [ ${#NODES[@]} -lt 2 ]; then
    echo "Error: need at least 2 node IPs"
    exit 1
fi

ec2_ssh() {
    local ip="$1"; shift
    ssh -i "$EC2_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        "${SSH_USER}@${ip}" "$@"
}

ec2_scp() {
    scp -i "$EC2_KEY" -o StrictHostKeyChecking=no "$@"
}

case "$ACTION" in
    setup-ssh)
        echo "=== Setting up passwordless SSH for MPI across ${#NODES[@]} nodes ==="

        # Generate MPI key pair on master (this machine)
        if [ ! -f "$MPI_KEY" ]; then
            ssh-keygen -t rsa -b 4096 -f "$MPI_KEY" -N "" -q
            echo "Generated MPI key: $MPI_KEY"
        else
            echo "MPI key already exists: $MPI_KEY"
        fi

        PUBKEY=$(cat "${MPI_KEY}.pub")

        for ip in "${NODES[@]}"; do
            echo "  Configuring $ip ..."

            # Copy MPI private key + public key to node
            ec2_scp "$MPI_KEY" "${SSH_USER}@${ip}:~/.ssh/mpi_cluster_rsa"
            ec2_scp "${MPI_KEY}.pub" "${SSH_USER}@${ip}:~/.ssh/mpi_cluster_rsa.pub"

            # Add public key to authorized_keys, set permissions
            ec2_ssh "$ip" bash -s <<SETUP
chmod 600 ~/.ssh/mpi_cluster_rsa
chmod 644 ~/.ssh/mpi_cluster_rsa.pub

# Add MPI public key if not already there
grep -qF "$PUBKEY" ~/.ssh/authorized_keys 2>/dev/null || \
    echo "$PUBKEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
SETUP

            # Configure SSH to use MPI key for all cluster nodes
            # Build ssh config for all nodes
            SSH_CONFIG=""
            for peer_ip in "${NODES[@]}"; do
                SSH_CONFIG="${SSH_CONFIG}
Host ${peer_ip}
    IdentityFile ~/.ssh/mpi_cluster_rsa
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    LogLevel ERROR
    User ${SSH_USER}
"
            done

            ec2_ssh "$ip" "cat > ~/.ssh/config << 'SSHEOF'
${SSH_CONFIG}
SSHEOF
chmod 600 ~/.ssh/config"

            echo "    OK"
        done

        echo ""
        echo "SSH setup complete. Run: $0 test ${NODES[*]}"
        ;;

    create-hostfile)
        echo "=== Creating hostfile: $HOSTFILE ==="
        > "$HOSTFILE"
        for ip in "${NODES[@]}"; do
            echo "${ip} slots=${SLOTS_PER_NODE}" >> "$HOSTFILE"
        done
        cat "$HOSTFILE"

        # Also copy hostfile to all nodes
        for ip in "${NODES[@]}"; do
            ec2_scp "$HOSTFILE" "${SSH_USER}@${ip}:~/hostfile"
        done
        echo ""
        echo "Hostfile created and distributed to all nodes."
        ;;

    test)
        echo "=== Testing MPI connectivity across ${#NODES[@]} nodes ==="

        # First test SSH from node 0 to all others
        echo "[1/3] SSH connectivity from ${NODES[0]}:"
        for ip in "${NODES[@]}"; do
            result=$(ec2_ssh "${NODES[0]}" "ssh ${ip} hostname" 2>&1) || true
            echo "  ${NODES[0]} -> ${ip}: ${result}"
        done

        # Test MPI hello world
        echo ""
        echo "[2/3] MPI hello world (${#NODES[@]} nodes, $((${#NODES[@]} * SLOTS_PER_NODE)) total slots):"
        TOTAL_RANKS=${#NODES[@]}
        ec2_ssh "${NODES[0]}" "mpirun -np ${TOTAL_RANKS} --hostfile ~/hostfile \
            --map-by node --bind-to none \
            python3 -c '
from mpi4py import MPI
import socket
comm = MPI.COMM_WORLD
print(f\"Rank {comm.Get_rank()}/{comm.Get_size()} on {socket.gethostname()}\")
'" 2>&1

        # Test with all slots
        echo ""
        echo "[3/3] Full slot test ($((${#NODES[@]} * SLOTS_PER_NODE)) ranks):"
        TOTAL_SLOTS=$((${#NODES[@]} * SLOTS_PER_NODE))
        ec2_ssh "${NODES[0]}" "mpirun -np ${TOTAL_SLOTS} --hostfile ~/hostfile \
            --map-by node --bind-to none \
            hostname" 2>&1 | sort | uniq -c

        echo ""
        echo "MPI cluster is operational."
        ;;

    *)
        echo "Usage: $0 {setup-ssh|create-hostfile|test} <ip1> <ip2> ..."
        exit 1
        ;;
esac
