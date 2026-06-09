#!/usr/bin/env bash
# benchmark_service.sh — Install and control the resumable benchmark service.
#
# Run on the MASTER node after cluster setup.
#
# Usage:
#   ./benchmark_service.sh install      Install + enable the systemd service
#   ./benchmark_service.sh start        Start the benchmark (or resume after crash)
#   ./benchmark_service.sh stop         Stop the benchmark gracefully
#   ./benchmark_service.sh status       Show service status + WAL state + logs
#   ./benchmark_service.sh logs         Tail the live benchmark log
#   ./benchmark_service.sh uninstall    Remove the service
#
# How it works:
#   systemd runs run_mpi_experiments.sh with Restart=on-failure.
#   If mpirun crashes → script's retry loop restarts it (Layer 2).
#   If the script itself crashes → systemd restarts it (Layer 3).
#   On each restart, the WAL tells run() which steps already
#   completed → simulation resumes, no work is lost (Layer 1).

set -euo pipefail

SERVICE_NAME="wenbo-benchmark"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_FILE="/mnt/nvme/wenbo_data/benchmark.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="/mnt/nvme/wenbo_data/results"
BENCH_DIR="/mnt/nvme/wenbo_data/mpi_38q_reord"

if [ $# -lt 1 ]; then
    echo "Usage: $0 {install|start|stop|status|logs|uninstall}"
    exit 1
fi

ACTION="$1"

case "$ACTION" in
    install)
        echo "Installing ${SERVICE_NAME} service..."

        # Ensure log directory exists
        sudo mkdir -p /mnt/nvme/wenbo_data

        # Copy service file
        sudo cp "${SCRIPT_DIR}/wenbo-benchmark.service" "$SERVICE_FILE"
        sudo systemctl daemon-reload
        sudo systemctl enable "${SERVICE_NAME}"

        echo "Service installed and enabled."
        echo ""
        echo "Commands:"
        echo "  $0 start     Start the benchmark"
        echo "  $0 status    Check progress"
        echo "  $0 logs      Watch live output"
        echo "  $0 stop      Stop gracefully"
        ;;

    start)
        if systemctl is-active --quiet "${SERVICE_NAME}"; then
            echo "Service is already running."
            echo "Use '$0 status' to check progress, or '$0 logs' to watch."
            exit 0
        fi

        echo "Starting ${SERVICE_NAME}..."

        # Check prerequisites
        if [ ! -f /home/ubuntu/hostfile ]; then
            echo "ERROR: /home/ubuntu/hostfile not found."
            echo "Run mpi_cluster.sh create-hostfile first."
            exit 1
        fi

        sudo systemctl start "${SERVICE_NAME}"
        echo "Benchmark started."
        echo ""
        echo "  Watch logs:    $0 logs"
        echo "  Check status:  $0 status"
        echo ""
        echo "You can safely disconnect SSH. The benchmark will:"
        echo "  - Continue running after disconnect"
        echo "  - Auto-restart on crash (up to 10x/hour)"
        echo "  - Resume from WAL checkpoint on each restart"
        echo "  - Stop cleanly when finished"
        ;;

    stop)
        echo "Stopping ${SERVICE_NAME}..."
        sudo systemctl stop "${SERVICE_NAME}"
        echo "Service stopped. WAL state is preserved."
        echo "Run '$0 start' to resume from where it left off."
        ;;

    status)
        echo "============================================================"
        echo "  ${SERVICE_NAME} — Status"
        echo "============================================================"
        echo ""

        # Systemd status
        echo "  Service:"
        if systemctl is-active --quiet "${SERVICE_NAME}"; then
            echo "    state: RUNNING"
            STARTED=$(systemctl show "${SERVICE_NAME}" --property=ActiveEnterTimestamp --value)
            echo "    since: ${STARTED}"
            RESTARTS=$(systemctl show "${SERVICE_NAME}" --property=NRestarts --value 2>/dev/null || echo "?")
            echo "    restarts: ${RESTARTS}"
        elif systemctl is-failed --quiet "${SERVICE_NAME}"; then
            echo "    state: FAILED"
            RESTARTS=$(systemctl show "${SERVICE_NAME}" --property=NRestarts --value 2>/dev/null || echo "?")
            echo "    restarts: ${RESTARTS}"
        else
            echo "    state: STOPPED"
        fi
        echo ""

        # Result status
        echo "  Results:"
        if [ -f "${RESULTS_DIR}/mpi_38q_reord.json" ]; then
            echo "    38q benchmark: COMPLETED"
        elif [ -f "${RESULTS_DIR}/smoke_test.json" ]; then
            PASS=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/smoke_test.json'))['pass'])" 2>/dev/null || echo "?")
            echo "    smoke test: ${PASS}"
            echo "    38q benchmark: IN PROGRESS"
        else
            echo "    smoke test: NOT RUN"
            echo "    38q benchmark: NOT STARTED"
        fi
        echo ""

        # WAL state
        echo "  WAL checkpoints:"
        for r in 0 1 2 3; do
            WAL_FILE="${BENCH_DIR}/rank_${r}/wal.json"
            if [ -f "$WAL_FILE" ]; then
                python3 -c "
import json
w = json.load(open('${WAL_FILE}'))
print(f'    rank {$r}: step {w[\"done_steps\"]}/5, buf={w[\"committed_buf\"]}')
"
            else
                echo "    rank ${r}: no checkpoint"
            fi
        done
        echo ""

        # Disk
        echo "  Disk:"
        df -h /mnt/nvme 2>/dev/null | tail -1 | awk '{print "    " $3 " used / " $2 " total (" $5 ")"}'
        echo ""

        # Last few log lines
        if [ -f "$LOG_FILE" ]; then
            echo "  Recent log:"
            tail -5 "$LOG_FILE" | sed 's/^/    /'
        fi

        echo ""
        echo "============================================================"
        ;;

    logs)
        if [ ! -f "$LOG_FILE" ]; then
            echo "No log file yet. Start the benchmark first."
            exit 1
        fi
        echo "Tailing ${LOG_FILE} (Ctrl+C to stop)..."
        echo ""
        tail -f "$LOG_FILE"
        ;;

    uninstall)
        echo "Uninstalling ${SERVICE_NAME}..."
        sudo systemctl stop "${SERVICE_NAME}" 2>/dev/null || true
        sudo systemctl disable "${SERVICE_NAME}" 2>/dev/null || true
        sudo rm -f "$SERVICE_FILE"
        sudo systemctl daemon-reload
        echo "Service removed. WAL and data are preserved on disk."
        ;;

    *)
        echo "Usage: $0 {install|start|stop|status|logs|uninstall}"
        exit 1
        ;;
esac
