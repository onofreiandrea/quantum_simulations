"""Quick script to check clean memory test results"""
import sys
from pathlib import Path
import glob

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyspark.sql import SparkSession

spark = SparkSession.builder.master('local[1]').appName('check').getOrCreate()

print("="*70)
print("CHECKING CLEAN MEMORY TEST RESULTS")
print("="*70)

results = {}
for n in range(25, 31):
    base_path = Path(f"data/clean_test_RY_gates_{n}")
    
    # Check for state files
    state_files = list(base_path.glob("state/**/*.parquet"))
    checkpoint_files = list(base_path.glob("checkpoints/**/*.parquet"))
    metadata_exists = (base_path / "metadata.duckdb").exists()
    
    if state_files:
        try:
            # Read ALL parquet files in the state directory (not just one)
            state_dir = state_files[0].parent
            df = spark.read.parquet(str(state_dir))
            count = df.count()
            theoretical = 2 ** n
            sparsity = (1 - count / theoretical) * 100
            results[n] = ('SUCCESS', count, sparsity)
        except Exception as e:
            results[n] = ('ERROR', str(e)[:60], None)
    elif checkpoint_files:
        try:
            # Read ALL parquet files in the checkpoint directory
            checkpoint_dir = checkpoint_files[0].parent
            df = spark.read.parquet(str(checkpoint_dir))
            count = df.count()
            theoretical = 2 ** n
            sparsity = (1 - count / theoretical) * 100
            results[n] = ('CHECKPOINT_ONLY', count, sparsity)
        except Exception as e:
            results[n] = ('ERROR', str(e)[:60], None)
    elif metadata_exists:
        results[n] = ('INCOMPLETE', 'Metadata exists but no state', None)
    else:
        results[n] = ('NOT_FOUND', None, None)

print("\nResults:")
print("-"*70)
for n in sorted(results.keys()):
    status, data, sparsity = results[n]
    if status == 'SUCCESS':
        print(f"{n} qubits: ✅ SUCCESS - {data:,} amplitudes ({sparsity:.1f}% sparse)")
    elif status == 'CHECKPOINT_ONLY':
        print(f"{n} qubits: ⚠️  CHECKPOINT ONLY - {data:,} amplitudes ({sparsity:.1f}% sparse)")
    elif status == 'INCOMPLETE':
        print(f"{n} qubits: ⏸️  INCOMPLETE - {data}")
    elif status == 'ERROR':
        print(f"{n} qubits: ❌ ERROR - {data}")
    else:
        print(f"{n} qubits: ❌ NOT FOUND")

max_success = max([n for n, (s, _, _) in results.items() if s in ['SUCCESS', 'CHECKPOINT_ONLY']], default=None)
if max_success:
    print(f"\n🎯 MAX SUCCESS: {max_success} qubits")
else:
    print("\n⚠️  No successful tests found")

spark.stop()
