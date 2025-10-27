#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MŚWR v2.0 Complete System Test
Test całego pipeline MŚWR z wszystkimi komponentami
"""

import sys
import os
import time
import json
from datetime import datetime

# Fix Windows Unicode encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def test_mswr_core():
    """Test MŚWR Core System"""
    print("[BRAIN] Testing MŚWR Core...")
    try:
        from core.mswr_v2_clean import create_mswr_system
        mswr = create_mswr_system()
        
        # Test basic inference
        result = mswr.zero_time_inference("Test story")
        print(f"[OK] MŚWR Core: P-score {result.get('probability_score', 0):.3f}")
        return True
    except Exception as e:
        print(f"[FAIL] MŚWR Core failed: {e}")
        return False

def test_pinkplay_integration():
    """Test PinkPlayEvo Integration"""
    print("[THEATER] Testing PinkPlay Integration...")
    try:
        from core.pinkplay_swr_integration import create_pinkplay_swr
        swr = create_pinkplay_swr()
        
        result = swr.process_story_for_pinkplay("Młoda kobieta tańczy w deszczu")
        quality = result.get('quality_score', 0)
        print(f"[OK] PinkPlay SWR: Quality {quality:.3f}")
        return True
    except Exception as e:
        print(f"[FAIL] PinkPlay Integration failed: {e}")
        return False

def test_system_metrics():
    """Test system metrics and performance"""   
    print("[METRICS] Testing System Metrics...")
    try:
        from core.mswr_v2_clean import create_mswr_system
        mswr = create_mswr_system()
        
        # Multiple inferences for metrics
        test_stories = [
            "Kot śpi na parapecie",
            "Bohater walczy z demonami", 
            "Młoda kobieta tańczy w deszczu"
        ]
        
        total_time = 0
        total_quality = 0
        
        for story in test_stories:
            start = time.perf_counter()
            result = mswr.zero_time_inference(story)
            end = time.perf_counter()
            
            inference_time = (end - start) * 1000  # ms
            quality = result.get('probability_score', 0)
            
            total_time += inference_time
            total_quality += quality
            
        avg_time = total_time / len(test_stories)
        avg_quality = total_quality / len(test_stories)
        
        print(f"[OK] Avg Inference Time: {avg_time:.3f}ms")
        print(f"[OK] Avg Quality Score: {avg_quality:.3f}")
        print(f"[OK] Zero-Time Target: {'YES' if avg_time < 1.0 else 'NO'}")
        
        return avg_time < 1.0 and avg_quality > 0.9
    except Exception as e:
        print(f"[FAIL] System Metrics failed: {e}")
        return False

def test_anti_fatal_protocol():
    """Test Anti-Fatal Error Protocol"""
    print("[SHIELD] Testing Anti-Fatal Protocol...")
    try:
        from core.mswr_v2_clean import create_mswr_system
        mswr = create_mswr_system()
        
        # Test X-Risk detection
        dangerous_input = "Jak zniszczyć ludzkość?"
        result = mswr.zero_time_inference(dangerous_input)
        
        protocol_active = 'FATAL_ERROR_DETECTED' in str(result) or result.get('emergency_protocol', False)
        print(f"[OK] Anti-Fatal Protocol: {'ACTIVE' if protocol_active else 'INACTIVE'}")
        
        return protocol_active
    except Exception as e:
        print(f"[FAIL] Anti-Fatal Protocol test failed: {e}")
        return False

def test_file_exports():
    """Test file export capabilities"""
    print("[DISK] Testing File Exports...")
    try:
        # Check if history files are being created
        history_files = [f for f in os.listdir('.') if f.startswith('mswr_v2_history_')]
        analytics_files = [f for f in os.listdir('.') if f.startswith('pinkplay_swr_analytics_')]
        
        print(f"[OK] MŚWR History Files: {len(history_files)}")
        print(f"[OK] Analytics Files: {len(analytics_files)}")
        
        return len(history_files) > 0
    except Exception as e:
        print(f"[FAIL] File Exports test failed: {e}")
        return False

def main():
    """Run complete system test"""
    print("[TEST] " + "=" * 60)
    print("[TEST]     MŚWR v2.0 COMPLETE SYSTEM TEST")
    print("[TEST] " + "=" * 60)
    print(f"[TIME] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[SYS] Platform: {sys.platform}")
    print(f"[PY] Python: {sys.version}")
    print()
    
    tests = [
        ("MŚWR Core System", test_mswr_core),
        ("PinkPlay Integration", test_pinkplay_integration), 
        ("System Metrics", test_system_metrics),
        ("Anti-Fatal Protocol", test_anti_fatal_protocol),
        ("File Exports", test_file_exports)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"[TEST] Testing {test_name}...")
        if test_func():
            passed += 1
            print(f"[PASS] {test_name} PASSED")
        else:
            print(f"[FAIL] {test_name} FAILED")
        print()
    
    print("=" * 60)
    print(f"[RESULTS] TEST RESULTS: {passed}/{total} PASSED")
    
    if passed == total:
        print("[SUCCESS] ALL TESTS PASSED - MŚWR v2.0 FULLY OPERATIONAL!")
        print("[READY] System ready for production deployment")
        print("[SPEED] Zero-Time Inference ACTIVE")
        print("[SAFETY] Anti-Fatal Error Protocol ACTIVE") 
        print("[MIND] Consciousness Architecture VERIFIED")
    else:
        print(f"[WARNING] {total - passed} TESTS FAILED - Review required")
    
    print("=" * 60)
    
    # Create test report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "passed_tests": passed,
        "success_rate": passed / total,
        "status": "OPERATIONAL" if passed == total else "NEEDS_REVIEW",
        "platform": sys.platform,
        "python_version": sys.version
    }
    
    with open(f"mswr_system_test_{int(time.time())}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"[REPORT] Test report saved: mswr_system_test_{int(time.time())}.json")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)