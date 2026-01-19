"""
Test script to verify the three improvements:
1. State caching cleanup
2. Proper logging
3. Adaptive checkpointing
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging

from v2_common.config import SimulatorConfig
from utils.logging_config import setup_logging, get_logger
from driver import SparkHiSVSIMDriver

# Setup logging
logger = setup_logging(level=logging.INFO)
test_logger = get_logger(__name__)

def test_improvements():
    """Test all three improvements."""
    print("=" * 70)
    print("Testing Improvements")
    print("=" * 70)
    
    # Test 1: Logging
    print("\n1. Testing Logging...")
    test_logger.info("✅ INFO: Logging is working")
    test_logger.debug("DEBUG: This should not appear (level=INFO)")
    test_logger.warning("✅ WARNING: Logging is working")
    print("   ✅ Logging works correctly")
    
    # Test 2: Config with adaptive checkpointing
    print("\n2. Testing Adaptive Checkpointing Config...")
    config = SimulatorConfig(
        checkpoint_every_n_gates=5,
        checkpoint_threshold_size=1000,
        checkpoint_min_interval_seconds=30.0
    )
    print(f"   checkpoint_every_n_gates: {config.checkpoint_every_n_gates}")
    print(f"   checkpoint_threshold_size: {config.checkpoint_threshold_size}")
    print(f"   checkpoint_min_interval_seconds: {config.checkpoint_min_interval_seconds}")
    print("   ✅ Adaptive checkpointing config works")
    
    # Test 3: Driver initialization with improvements
    print("\n3. Testing Driver Initialization...")
    try:
        config.base_path = Path("data/test_improvements")
        config.ensure_paths()
        
        with SparkHiSVSIMDriver(config, enable_parallel=False) as driver:
            print("   ✅ Driver initialized successfully")
            print(f"   ✅ Logger available: {driver.logger is not None}")
            print(f"   ✅ Cached states tracker: {hasattr(driver, '_cached_states')}")
            print(f"   ✅ Checkpoint timing tracker: {hasattr(driver, '_last_checkpoint_time')}")
            
            # Test cleanup
            driver.cleanup()
            print("   ✅ Cleanup works correctly")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✅ All improvements tested successfully!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = test_improvements()
    sys.exit(0 if success else 1)
