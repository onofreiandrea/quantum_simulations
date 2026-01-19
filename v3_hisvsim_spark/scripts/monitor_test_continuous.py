"""Continuously monitor comprehensive max test progress"""
import time
import json
from pathlib import Path
import sys

def check_test_status():
    results_file = Path("data/comprehensive_non_stabilizer_max_results.json")
    log_file = Path("/tmp/comprehensive_max_clean.log")
    
    # Check if test is running
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        is_running = "test_all_non_stabilizer_max_clean" in result.stdout or "docker" in result.stdout.lower()
    except:
        is_running = False
    
    # Check log file
    log_lines = []
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                log_lines = lines[-20:] if len(lines) > 20 else lines
        except:
            pass
    
    # Check results file
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
            
            print("\n" + "="*70)
            print("COMPREHENSIVE TEST - CURRENT STATUS")
            print("="*70)
            print(f"Test started: {data.get('timestamp', 'Unknown')}")
            print(f"Strategy: {data.get('strategy', 'Unknown')}")
            print(f"Test running: {'✅ YES' if is_running else '❌ NO'}")
            print()
            
            results = data.get('results', {})
            detailed = data.get('detailed_results', {})
            
            print("Results Summary:")
            print("-"*70)
            for name in ["RY gates", "H+T gates", "H+T+CR gates", "G gates", "R gates", "CU gates"]:
                max_q = results.get(name)
                if max_q:
                    print(f"{name:20s}: {max_q:2d} qubits ✅")
                    if name in detailed:
                        details = detailed[name]
                        successes = [int(q) for q, d in details.items() if isinstance(d, dict) and d.get('status') == 'SUCCESS']
                        failures = [int(q) for q, d in details.items() if isinstance(d, dict) and d.get('status') == 'FAILED']
                        if successes:
                            last_success = max(successes)
                            last_detail = details[str(last_success)]
                            amps = last_detail.get('amplitudes', 0)
                            time_sec = last_detail.get('elapsed_seconds', 0)
                            print(f"  └─ Last success: {last_success} qubits ({amps:,} amplitudes, {time_sec:.1f}s)")
                        if failures:
                            first_failure = min(failures)
                            fail_detail = details[str(first_failure)]
                            error_type = fail_detail.get('error_type', 'UNKNOWN')
                            print(f"  └─ First failure: {first_failure} qubits ({error_type})")
                elif max_q is None and name in detailed:
                    # Test started but no success yet
                    details = detailed[name]
                    in_progress = [int(q) for q, d in details.items() if isinstance(d, dict)]
                    if in_progress:
                        print(f"{name:20s}: Testing... (checked up to {max(in_progress)} qubits)")
                    else:
                        print(f"{name:20s}: Not started yet")
                else:
                    print(f"{name:20s}: Not tested yet")
            
            overall_max = data.get('overall_max', 0)
            print()
            print(f"🎯 OVERALL MAX: {overall_max} qubits")
            
            # Show recent log activity
            if log_lines:
                print()
                print("Recent log activity:")
                print("-"*70)
                for line in log_lines[-10:]:
                    if any(keyword in line for keyword in ["Testing:", "qubits", "SUCCESS", "FAILED", "MAX"]):
                        print(line.strip()[:100])
            
            return True
        except Exception as e:
            print(f"Error reading results: {e}")
            return False
    else:
        print("\n" + "="*70)
        print("COMPREHENSIVE TEST - STATUS")
        print("="*70)
        print(f"Test running: {'✅ YES' if is_running else '❌ NO'}")
        print("Results file: Not created yet")
        print("Status: Test may be initializing...")
        
        if log_lines:
            print()
            print("Recent log activity:")
            print("-"*70)
            for line in log_lines[-10:]:
                print(line.strip()[:100])
        
        return False

if __name__ == "__main__":
    import signal
    
    def signal_handler(sig, frame):
        print("\n\nMonitoring stopped.")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    print("Starting continuous monitoring...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            check_test_status()
            print("\n" + "="*70)
            print("Next check in 30 seconds... (Ctrl+C to stop)")
            print("="*70)
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)
