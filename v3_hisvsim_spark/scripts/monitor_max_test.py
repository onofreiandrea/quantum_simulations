"""Monitor comprehensive max test progress"""
import json
from pathlib import Path
import sys

results_file = Path("data/comprehensive_non_stabilizer_max_results.json")

if results_file.exists():
    with open(results_file) as f:
        data = json.load(f)
    
    print("="*70)
    print("COMPREHENSIVE NON-STABILIZER MAX TEST - CURRENT RESULTS")
    print("="*70)
    print(f"Test started: {data.get('timestamp', 'Unknown')}")
    print(f"Strategy: {data.get('strategy', 'Unknown')}")
    print()
    
    results = data.get('results', {})
    detailed = data.get('detailed_results', {})
    
    print("Results Summary:")
    print("-"*70)
    for name, max_q in sorted(results.items()):
        if max_q:
            print(f"{name:20s}: {max_q:2d} qubits ✅")
            # Show details for this gate type
            if name in detailed:
                details = detailed[name]
                successes = [q for q, d in details.items() if d.get('status') == 'SUCCESS']
                if successes:
                    last_success = max(successes)
                    last_detail = details[last_success]
                    amps = last_detail.get('amplitudes', 0)
                    time = last_detail.get('elapsed_seconds', 0)
                    print(f"  └─ Last success: {last_success} qubits ({amps:,} amplitudes, {time:.1f}s)")
        else:
            print(f"{name:20s}: Failed ❌")
    
    overall_max = data.get('overall_max', 0)
    print()
    print(f"🎯 OVERALL MAX: {overall_max} qubits")
    
else:
    print("Results file not found yet. Test may still be running.")
    print(f"Expected location: {results_file}")
