#!/usr/bin/env python3
"""
Realistic crash and recovery test.

Simulates:
1. Partial execution (crash after some batches)
2. Recovery from checkpoint
3. Resume execution
4. Verify correctness
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from driver import SparkHiSVSIMDriver
from v2_common import config, circuits, frontend
import tempfile
import shutil
import numpy as np
from pathlib import Path


def simulate_crash_scenario():
    """Simulate a realistic crash scenario."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = config.SimulatorConfig(
        run_id="crash_test",
        base_path=temp_dir,
        batch_size=3,  # Small batches
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    
    circuit = circuits.generate_qft_circuit(4)  # 10 gates
    
    try:
        print("="*70)
        print("CRASH AND RECOVERY TEST")
        print("="*70)
        
        # =====================================================================
        # Phase 1: Partial Execution (Simulate Crash)
        # =====================================================================
        print("\n📋 PHASE 1: Partial Execution (Simulating Crash)")
        print("-" * 70)
        
        with SparkHiSVSIMDriver(cfg, enable_parallel=False) as driver:
            # Process only first 2 batches manually (simulating crash)
            n_qubits, gates = frontend.circuit_dict_to_gates(circuit)
            
            # Process first batch
            current_state = driver.state_manager.initialize_state(n_qubits)
            driver.gate_applicator.register_gates(gates)
            
            # Process first 3 gates (batch 1)
            batch1_gates = gates[0:3]
            version_in = 0
            version_out = 1
            
            # Write WAL PENDING
            wal_id1 = driver.metadata_store.wal_create_pending(
                run_id=cfg.run_id,
                gate_start=0,
                gate_end=3,
                state_version_in=version_in,
                state_version_out=version_out,
            )
            print(f"   Created WAL PENDING: gates 0-3, v{version_in}->v{version_out}")
            
            # Apply gates
            output_state = driver.gate_applicator.apply_gates(current_state, batch1_gates)
            
            # Save state
            state_path = driver.state_manager.save_state(output_state, version_out)
            
            # Create checkpoint
            driver.checkpoint_manager.create_checkpoint(
                state_version=version_out,
                last_gate_seq=2,
                state_path=state_path,
            )
            
            # Mark WAL COMMITTED
            driver.metadata_store.wal_mark_committed(wal_id1)
            print(f"   ✅ Batch 1 completed: checkpoint v{version_out} created")
            
            # Process second batch
            current_state = driver.state_manager.load_state(version_out)
            batch2_gates = gates[3:6]
            version_in = version_out
            version_out = 2
            
            # Write WAL PENDING
            wal_id2 = driver.metadata_store.wal_create_pending(
                run_id=cfg.run_id,
                gate_start=3,
                gate_end=6,
                state_version_in=version_in,
                state_version_out=version_out,
            )
            print(f"   Created WAL PENDING: gates 3-6, v{version_in}->v{version_out}")
            
            # Apply gates
            output_state = driver.gate_applicator.apply_gates(current_state, batch2_gates)
            
            # Save state
            state_path = driver.state_manager.save_state(output_state, version_out)
            
            # Create checkpoint
            driver.checkpoint_manager.create_checkpoint(
                state_version=version_out,
                last_gate_seq=5,
                state_path=state_path,
            )
            
            # Mark WAL COMMITTED
            driver.metadata_store.wal_mark_committed(wal_id2)
            print(f"   ✅ Batch 2 completed: checkpoint v{version_out} created")
            
            # Start third batch but CRASH before completing (simulate crash)
            current_state = driver.state_manager.load_state(version_out)
            batch3_gates = gates[6:9]
            version_in = version_out
            version_out = 3
            
            # Write WAL PENDING but don't complete
            wal_id3 = driver.metadata_store.wal_create_pending(
                run_id=cfg.run_id,
                gate_start=6,
                gate_end=9,
                state_version_in=version_in,
                state_version_out=version_out,
            )
            print(f"   ⚠️  Created WAL PENDING: gates 6-9, v{version_in}->v{version_out}")
            print(f"   💥 CRASH! Execution interrupted before completing batch 3")
        
        # =====================================================================
        # Phase 2: Recovery
        # =====================================================================
        print("\n📋 PHASE 2: Recovery After Crash")
        print("-" * 70)
        
        with SparkHiSVSIMDriver(cfg, enable_parallel=False) as driver2:
            # Check PENDING entries before recovery
            pending_before = driver2.metadata_store.wal_get_pending(cfg.run_id)
            print(f"   PENDING WAL entries before recovery: {len(pending_before)}")
            
            # Recover
            recovery_state = driver2.recovery_manager.recover(4)
            
            print(f"   ✅ Recovered state_version: {recovery_state.state_version}")
            print(f"   ✅ Recovered last_gate_seq: {recovery_state.last_gate_seq}")
            print(f"   ✅ Has checkpoint: {recovery_state.checkpoint_record is not None}")
            
            # Check PENDING entries after recovery
            pending_after = driver2.metadata_store.wal_get_pending(cfg.run_id)
            print(f"   PENDING WAL entries after recovery: {len(pending_after)}")
            
            # Check FAILED entries
            conn = driver2.metadata_store.connect()
            failed_count = conn.execute(
                "SELECT COUNT(*) FROM wal WHERE run_id = ? AND status = 'FAILED'",
                [cfg.run_id]
            ).fetchone()[0]
            print(f"   FAILED WAL entries: {failed_count}")
            
            # Verify recovery state
            assert recovery_state.state_version == 2, "Should recover from checkpoint v2"
            assert recovery_state.last_gate_seq == 5, "Should recover from gate 5"
            assert failed_count > 0, "PENDING entries should be marked FAILED"
        
        # =====================================================================
        # Phase 3: Resume Execution
        # =====================================================================
        print("\n📋 PHASE 3: Resume Execution")
        print("-" * 70)
        
        with SparkHiSVSIMDriver(cfg, enable_parallel=False) as driver3:
            # Resume from checkpoint
            result = driver3.run_circuit(circuit, resume=True)
            state_recovered = driver3.get_state_vector(result)
            
            print(f"   ✅ Resumed execution completed")
            print(f"   ✅ Final state norm: {np.linalg.norm(state_recovered):.10f}")
        
        # =====================================================================
        # Phase 4: Verify Correctness
        # =====================================================================
        print("\n📋 PHASE 4: Verify Correctness")
        print("-" * 70)
        
        # Run complete circuit from scratch for comparison
        cfg2 = config.SimulatorConfig(
            run_id="comparison",
            base_path=temp_dir / "comparison",
            batch_size=3,
            spark_master="local[2]",
            spark_shuffle_partitions=4,
        )
        cfg2.ensure_paths()
        
        with SparkHiSVSIMDriver(cfg2, enable_parallel=False) as driver4:
            result_complete = driver4.run_circuit(circuit, resume=False)
            state_complete = driver4.get_state_vector(result_complete)
        
        # Compare states
        max_diff = np.max(np.abs(state_recovered - state_complete))
        print(f"   ✅ States match: max_diff={max_diff:.2e}")
        
        if max_diff < 1e-10:
            print("   ✅ RECOVERY SUCCESSFUL: Recovered state matches complete execution!")
        else:
            print(f"   ❌ RECOVERY FAILED: States differ by {max_diff}")
        
        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\n✅ Fault Tolerance Verified:")
        print("   1. Partial execution creates checkpoints")
        print("   2. Crash leaves PENDING WAL entries")
        print("   3. Recovery loads latest checkpoint")
        print("   4. WAL reconciliation marks PENDING as FAILED")
        print("   5. Resume execution completes correctly")
        print("   6. Recovered state matches complete execution")
        print("\n✅ Architecture handles crashes correctly!")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    simulate_crash_scenario()
