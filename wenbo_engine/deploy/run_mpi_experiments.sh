#!/usr/bin/env bash
# run_mpi_experiments.sh — Run MPI benchmark (smoke test + 38q reordered)
#
# Resumable: re-running after a crash will skip completed steps and
# resume the 38q benchmark from the last WAL-committed step.
#
# Usage:
#   ./run_mpi_experiments.sh --hostfile ~/hostfile
#   ./run_mpi_experiments.sh --hostfile ~/hostfile --work-dir /mnt/nvme/wenbo_data
#   ./run_mpi_experiments.sh --hostfile ~/hostfile --status   # check run status

set -euo pipefail

# ── Parse args ──────────────────────────────────────────────────────
HOSTFILE=""
WORK_DIR="/mnt/nvme/wenbo_data"
N_RANKS=4
CHUNK_BITS=24
STATUS_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --hostfile)   HOSTFILE="$2"; shift 2 ;;
        --work-dir)   WORK_DIR="$2"; shift 2 ;;
        --ranks)      N_RANKS="$2"; shift 2 ;;
        --chunk-bits) CHUNK_BITS="$2"; shift 2 ;;
        --status)     STATUS_ONLY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$HOSTFILE" ]; then
    echo "Usage: $0 --hostfile <path> [--work-dir <path>] [--ranks <n>] [--status]"
    exit 1
fi

RESULTS_DIR="${WORK_DIR}/results"
BENCH38_DIR="${WORK_DIR}/mpi_38q_reord"

MPI_OPTS=(
    -np "$N_RANKS"
    --hostfile "$HOSTFILE"
    --map-by node
    --bind-to none
    -x PYTHONPATH
    -x NUMBA_NUM_THREADS
    -x OMP_NUM_THREADS
    -x MKL_NUM_THREADS
    -x OPENBLAS_NUM_THREADS
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Let numba use all cores; disable BLAS threading so it doesn't
# compete with numba's prange thread pool during gate computation.
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
export NUMBA_NUM_THREADS="$NCPU"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Status check ────────────────────────────────────────────────────

check_status() {
    echo "============================================================"
    echo "  wenbo_engine MPI — Run Status"
    echo "============================================================"

    # Smoke test
    if [ -f "${RESULTS_DIR}/smoke_test.json" ]; then
        PASS=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/smoke_test.json'))['pass'])")
        if [ "$PASS" = "True" ]; then
            echo "  Smoke test:  PASSED"
        else
            echo "  Smoke test:  FAILED"
        fi
    else
        echo "  Smoke test:  NOT RUN"
    fi

    # 38q benchmark result
    if [ -f "${RESULTS_DIR}/mpi_38q_reord.json" ]; then
        echo "  38q benchmark: COMPLETED"
        python3 -c "
import json
r = json.load(open('${RESULTS_DIR}/mpi_38q_reord.json'))
for k, v in r.items():
    if isinstance(v, float):
        print(f'    {k}: {v:.6g}')
    else:
        print(f'    {k}: {v}')
"
    else
        echo "  38q benchmark: NOT COMPLETED"
    fi

    # WAL state per rank
    echo ""
    echo "  WAL state per rank:"
    for r in $(seq 0 $((N_RANKS - 1))); do
        WAL_FILE="${BENCH38_DIR}/rank_${r}/wal.json"
        if [ -f "$WAL_FILE" ]; then
            python3 -c "
import json
w = json.load(open('${WAL_FILE}'))
print(f'    rank {$r}: step {w[\"done_steps\"]}, buf={w[\"committed_buf\"]}')
"
        else
            echo "    rank ${r}: no WAL (not started)"
        fi
    done

    # Disk usage per rank
    echo ""
    echo "  Disk usage per rank:"
    for r in $(seq 0 $((N_RANKS - 1))); do
        RANK_DIR="${BENCH38_DIR}/rank_${r}"
        if [ -d "$RANK_DIR" ]; then
            SIZE=$(du -sh "$RANK_DIR" 2>/dev/null | cut -f1)
            echo "    rank ${r}: ${SIZE}"
        else
            echo "    rank ${r}: (not created)"
        fi
    done

    echo "============================================================"
}

if [ "$STATUS_ONLY" = true ]; then
    check_status
    exit 0
fi

# ── Main run ────────────────────────────────────────────────────────

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  wenbo_engine MPI Benchmark"
echo "============================================================"
echo "  Hostfile:    $HOSTFILE"
echo "  Ranks:       $N_RANKS"
echo "  Chunk bits:  $CHUNK_BITS"
echo "  Work dir:    $WORK_DIR"
echo "  Results:     $RESULTS_DIR"
echo "============================================================"
echo ""

# ── Step 0: Upload code to all nodes ─────────────────────────────
echo "[0/2] Distributing code to all nodes..."
NODES=($(awk '{print $1}' "$HOSTFILE"))
for ip in "${NODES[@]}"; do
    echo "  Syncing to $ip ..."
    rsync -az --exclude='__pycache__' --exclude='.git' --exclude='*.zip' \
        "${PROJECT_DIR}/" "ubuntu@${ip}:~/quantum_simulations/" 2>/dev/null || \
    scp -r -o StrictHostKeyChecking=no \
        "${PROJECT_DIR}/wenbo_engine" "ubuntu@${ip}:~/quantum_simulations/wenbo_engine"
done
echo "  Code distributed."
echo ""

# ── Step 1: Smoke test (skip if already passed) ─────────────────
SMOKE_PASSED=false
if [ -f "${RESULTS_DIR}/smoke_test.json" ]; then
    PREV_PASS=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/smoke_test.json'))['pass'])" 2>/dev/null || echo "False")
    if [ "$PREV_PASS" = "True" ]; then
        echo "[1/2] Smoke test — SKIPPED (already passed)"
        SMOKE_PASSED=true
    fi
fi

if [ "$SMOKE_PASSED" = false ]; then
    echo "[1/2] Smoke test (10q, ${N_RANKS} ranks)"
    SMOKE_DIR="${WORK_DIR}/smoke_test"
    SMOKE_START=$(date +%s)

    mpirun "${MPI_OPTS[@]}" \
        python3 -c "
import sys; sys.path.insert(0, '$HOME/quantum_simulations')
import json, numpy as np, tempfile
from mpi4py import MPI
from wenbo_engine.mpi.mpi_runner import run, collect_state, compute_norm
from wenbo_engine.tests.fixtures.circuits import quest_random
from wenbo_engine.kernel.ref_dense import simulate

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cd = quest_random(10, n_gates=10, seed=42)
wd = '${SMOKE_DIR}'
run(cd, wd, chunk_size=256, use_wal=True, comm=comm)
got = collect_state(wd, comm)
norm = compute_norm(wd, comm)
if rank == 0:
    ref = simulate(cd)
    err = float(np.max(np.abs(got - ref)))
    norm_err = float(abs(norm - 1.0))
    print(f'  max error:  {err:.2e}')
    print(f'  norm error: {norm_err:.2e}')
    print(f'  status:     {\"PASS\" if err < 1e-6 and norm_err < 1e-6 else \"FAIL\"}')
    json.dump({'max_error': err, 'norm_error': norm_err, 'pass': err < 1e-6},
              open('${RESULTS_DIR}/smoke_test.json', 'w'), indent=2)
" 2>&1

    SMOKE_END=$(date +%s)
    echo "  time: $((SMOKE_END - SMOKE_START))s"
    echo ""

    # Check smoke test passed before continuing
    if [ -f "${RESULTS_DIR}/smoke_test.json" ]; then
        SMOKE_PASS=$(python3 -c "import json; print(json.load(open('${RESULTS_DIR}/smoke_test.json'))['pass'])")
        if [ "$SMOKE_PASS" != "True" ]; then
            echo "SMOKE TEST FAILED — aborting."
            exit 1
        fi
    else
        echo "SMOKE TEST RESULTS NOT FOUND — aborting."
        exit 1
    fi
fi
echo ""

# ── Step 2: MPI 38q benchmark (50 gates, with reorder) ──────────
# Automatic recovery: if mpirun crashes, the WAL records which steps
# completed. The retry loop restarts mpirun, which calls run(), which
# reads the WAL and skips already-committed steps.  This is the core
# "resumable simulation" feature of the thesis.
MAX_RETRIES=5
RETRY_DELAY=10  # seconds between retries

if [ -f "${RESULTS_DIR}/mpi_38q_reord.json" ]; then
    echo "[2/2] MPI 38q benchmark — SKIPPED (already completed)"
    echo ""
else
    BENCH38_START=$(date +%s)
    ATTEMPT=0
    BENCH_SUCCESS=false

    while [ "$ATTEMPT" -lt "$MAX_RETRIES" ]; do
        ATTEMPT=$((ATTEMPT + 1))

        # Show WAL state if resuming
        WAL_EXISTS=false
        for r in $(seq 0 $((N_RANKS - 1))); do
            WAL_FILE="${BENCH38_DIR}/rank_${r}/wal.json"
            if [ -f "$WAL_FILE" ]; then
                WAL_EXISTS=true
                break
            fi
        done

        if [ "$WAL_EXISTS" = true ]; then
            echo "[2/2] MPI 38q benchmark — attempt ${ATTEMPT}/${MAX_RETRIES}, RESUMING from WAL"
            for r in $(seq 0 $((N_RANKS - 1))); do
                WAL_FILE="${BENCH38_DIR}/rank_${r}/wal.json"
                if [ -f "$WAL_FILE" ]; then
                    python3 -c "
import json
w = json.load(open('${WAL_FILE}'))
print(f'  rank ${r}: resuming from step {w[\"done_steps\"]}')
"
                fi
            done
        else
            echo "[2/2] MPI 38q benchmark (attempt ${ATTEMPT}/${MAX_RETRIES})"
        fi

        if mpirun "${MPI_OPTS[@]}" \
            python3 -m wenbo_engine.mpi.mpi_benchmark \
            --qubits 38 --gates 50 --seed 42 \
            --chunk-bits "$CHUNK_BITS" \
            --work-dir "$BENCH38_DIR" \
            --output "${RESULTS_DIR}/mpi_38q_reord.json" \
            2>&1; then
            BENCH_SUCCESS=true
            break
        else
            echo ""
            echo "  *** mpirun exited with error (attempt ${ATTEMPT}/${MAX_RETRIES}) ***"
            if [ "$ATTEMPT" -lt "$MAX_RETRIES" ]; then
                echo "  WAL ensures no work is lost. Retrying in ${RETRY_DELAY}s..."
                sleep "$RETRY_DELAY"
                # Double the delay for next retry (exponential backoff)
                RETRY_DELAY=$((RETRY_DELAY * 2))
            else
                echo "  All ${MAX_RETRIES} attempts failed. Run this script again to resume."
                echo "  WAL state is preserved — no work is lost."
                check_status
                exit 1
            fi
        fi
    done

    BENCH38_END=$(date +%s)
    echo "  wall time: $((BENCH38_END - BENCH38_START))s"
    echo ""
fi

# ── Summary ──────────────────────────────────────────────────────
echo "============================================================"
echo "  RESULTS"
echo "============================================================"

for f in smoke_test mpi_38q_reord; do
    RESULT_FILE="${RESULTS_DIR}/${f}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo ""
        echo "  --- ${f} ---"
        python3 -c "
import json
r = json.load(open('$RESULT_FILE'))
for k, v in r.items():
    if isinstance(v, float):
        print(f'    {k}: {v:.6g}')
    else:
        print(f'    {k}: {v}')
"
    else
        echo "  --- ${f} --- MISSING"
    fi
done

echo ""
echo "============================================================"
echo "  All results saved to: $RESULTS_DIR"
echo "============================================================"
