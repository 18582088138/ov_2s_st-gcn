#!/usr/bin/env python
"""
Test script for benchmark functionality
"""
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_benchmark_with_fake_data():
    """Test benchmark with generated fake data"""
    print("=" * 60)
    print("Test: Benchmark with Fake Data")
    print("=" * 60)

    # Check if model exists
    model_xml = 'export_models/test.xml'
    if not os.path.exists(model_xml):
        print("Model not found. Creating fake model...")
        os.system('python tools/create_fake_model.py --output fake_models/test.pt')
        os.system('python tools/export_to_openvino.py --weights fake_models/test.pt --output-dir export_models')

    # Test benchmark with auto-generated data
    print("\n--- Test 1: Auto-generated random data ---")
    os.system(f'python tools/openvino_e2e_pipeline.py --mode benchmark --model-xml {model_xml} --device CPU --num-iterations 20 --max-samples 5')

    print("\n✓ Test 1 completed")


def test_benchmark_with_saved_data():
    """Test benchmark with saved data file"""
    print("\n" + "=" * 60)
    print("Test: Benchmark with Saved Data")
    print("=" * 60)

    # Create test data
    test_data_dir = 'data/test'
    os.makedirs(test_data_dir, exist_ok=True)

    test_data_path = os.path.join(test_data_dir, 'benchmark_data.npy')
    if not os.path.exists(test_data_path):
        print(f"Creating test data: {test_data_path}")
        test_data = np.random.randn(10, 3, 300, 25, 2).astype(np.float32)
        np.save(test_data_path, test_data)
        print(f"✓ Created {test_data_path} with shape {test_data.shape}")

    model_xml = 'export_models/test.xml'

    # Test benchmark with saved data
    print("\n--- Test 2: Benchmark with saved data file ---")
    os.system(f'python tools/openvino_e2e_pipeline.py --mode benchmark --model-xml {model_xml} --data-path {test_data_path} --device CPU --num-iterations 30 --max-samples 5')

    print("\n✓ Test 2 completed")


def test_benchmark_gpu_vs_cpu():
    """Test benchmark comparing GPU vs CPU"""
    print("\n" + "=" * 60)
    print("Test: GPU vs CPU Benchmark")
    print("=" * 60)

    # Check GPU availability
    try:
        import openvino as ov
        core = ov.Core()
        has_gpu = any('GPU' in d for d in core.available_devices)
    except:
        has_gpu = False

    model_xml = 'export_models/test.xml'
    test_data_path = 'data/test/benchmark_data.npy'

    # CPU benchmark
    print("\n--- CPU Benchmark ---")
    os.system(f'python tools/openvino_e2e_pipeline.py --mode benchmark --model-xml {model_xml} --data-path {test_data_path} --device CPU --num-iterations 50 --max-samples 8 --output-dir outputs/benchmark_cpu')

    if has_gpu:
        # GPU benchmark
        print("\n--- GPU Benchmark ---")
        os.system(f'python tools/openvino_e2e_pipeline.py --mode benchmark --model-xml {model_xml} --data-path {test_data_path} --device GPU --num-iterations 50 --max-samples 8 --output-dir outputs/benchmark_gpu')

        # Compare results
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print("Check outputs/benchmark_cpu/benchmark_result.json")
        print("Check outputs/benchmark_gpu/benchmark_result.json")
    else:
        print("\n⏭️  Skipping GPU benchmark (no GPU available)")

    print("\n✓ Test 3 completed")


def test_max_samples_parameter():
    """Test that max_samples parameter works correctly"""
    print("\n" + "=" * 60)
    print("Test: max_samples Parameter")
    print("=" * 60)

    # Create test data with 20 samples
    test_data_dir = 'data/test'
    os.makedirs(test_data_dir, exist_ok=True)

    test_data_path = os.path.join(test_data_dir, 'max_samples_test.npy')
    print(f"Creating test data with 20 samples...")
    test_data = np.random.randn(20, 3, 300, 25, 2).astype(np.float32)
    np.save(test_data_path, test_data)
    print(f"✓ Created {test_data_path} with shape {test_data.shape}")

    model_xml = 'export_models/test.xml'

    # Test 1: Without max_samples (should use all 20)
    print("\n--- Test without max_samples (should use all 20 samples) ---")
    os.system(f'python tools/openvino_e2e_pipeline.py --mode benchmark --model-xml {model_xml} --data-path {test_data_path} --device CPU --num-iterations 40 --output-dir outputs/test_no_limit')

    # Test 2: With max_samples=5 (should use only 5)
    print("\n--- Test with max_samples=5 (should use only 5 samples) ---")
    os.system(f'python tools/openvino_e2e_pipeline.py --mode benchmark --model-xml {model_xml} --data-path {test_data_path} --device CPU --num-iterations 40 --max-samples 5 --output-dir outputs/test_limit_5')

    # Test 3: With max_samples=10 (should use only 10)
    print("\n--- Test with max_samples=10 (should use only 10 samples) ---")
    os.system(f'python tools/openvino_e2e_pipeline.py --mode benchmark --model-xml {model_xml} --data-path {test_data_path} --device CPU --num-iterations 40 --max-samples 10 --output-dir outputs/test_limit_10')

    print("\n✓ Test 4 completed")
    print("\nVerify the results:")
    print("  - outputs/test_no_limit/benchmark_result.json should show num_samples: 20")
    print("  - outputs/test_limit_5/benchmark_result.json should show num_samples: 5")
    print("  - outputs/test_limit_10/benchmark_result.json should show num_samples: 10")


def main():
    """Run all tests"""
    print("\n")
    print("=" * 60)
    print("OpenVINO E2E Pipeline Benchmark Test Suite")
    print("=" * 60)
    print()

    try:
        # Test 1: Benchmark with fake data
        test_benchmark_with_fake_data()

        # Test 2: Benchmark with saved data
        test_benchmark_with_saved_data()

        # Test 3: GPU vs CPU benchmark
        test_benchmark_gpu_vs_cpu()

        # Test 4: max_samples parameter
        test_max_samples_parameter()

        # Summary
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        print("\nYou can now use the benchmark mode to test inference latency:")
        print("\n  # Benchmark with auto-generated data")
        print("  python tools/openvino_e2e_pipeline.py \\")
        print("      --mode benchmark \\")
        print("      --model-xml export_models/test.xml \\")
        print("      --device GPU \\")
        print("      --num-iterations 100 \\")
        print("      --max-samples 10")
        print("\n  # Benchmark with real data")
        print("  python tools/openvino_e2e_pipeline.py \\")
        print("      --mode benchmark \\")
        print("      --model-xml export_models/test.xml \\")
        print("      --data-path data/ntu/xsub/val_data.npy \\")
        print("      --device GPU \\")
        print("      --num-iterations 200 \\")
        print("      --max-samples 50")
        print("\n" + "=" * 60)

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
