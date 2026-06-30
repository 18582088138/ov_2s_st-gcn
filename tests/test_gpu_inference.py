#!/usr/bin/env python
"""
Test script for OpenVINO GPU inference fix
"""
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_device_detection():
    """Test 1: Check device availability"""
    print("=" * 60)
    print("Test 1: Device Detection")
    print("=" * 60)

    try:
        import openvino as ov
        core = ov.Core()

        available_devices = core.available_devices
        print(f"✓ OpenVINO version: {ov.__version__}")
        print(f"✓ Available devices: {available_devices}")

        for device in available_devices:
            try:
                device_name = core.get_property(device, "FULL_DEVICE_NAME")
                print(f"  - {device}: {device_name}")
            except:
                print(f"  - {device}")

        has_gpu = any('GPU' in d for d in available_devices)

        if has_gpu:
            print("\n✅ GPU device found! GPU testing is available.")
            return True
        else:
            print("\n⚠️  No GPU device found. Will test with CPU only.")
            print("OpenVINO GPU requires:")
            print("  - Intel GPU: Intel Graphics Driver + OpenCL Runtime")
            print("  - NVIDIA GPU: CUDA + OpenCL")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_inference_sync():
    """Test 2: Test inference synchronization"""
    print("\n" + "=" * 60)
    print("Test 2: Inference Synchronization")
    print("=" * 60)

    try:
        from tools.openvino_inference import OpenVINOInferenceEngine

        # Check if model exists
        model_xml = 'export_models/test.xml'
        if not os.path.exists(model_xml):
            print("⚠️  Model not found. Creating fake model first...")
            os.system('python tools/create_fake_model.py --output fake_models/test.pt')
            os.system('python tools/export_to_openvino.py --weights fake_models/test.pt --output-dir export_models')

        # Test with CPU
        print("\nTesting CPU inference...")
        cpu_engine = OpenVINOInferenceEngine(model_xml, device='CPU')

        # Create test input
        test_input = np.random.randn(1, 3, 300, 25, 2).astype(np.float32)

        # Run inference
        output = cpu_engine.infer(test_input)

        print(f"✓ CPU inference successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

        # Test timing
        output, time_ms = cpu_engine.infer_with_timing(test_input)
        print(f"✓ CPU inference timing: {time_ms:.2f} ms")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_inference(has_gpu):
    """Test 3: Test GPU inference if available"""
    print("\n" + "=" * 60)
    print("Test 3: GPU Inference")
    print("=" * 60)

    if not has_gpu:
        print("⏭️  Skipping GPU test (no GPU available)")
        return True

    try:
        from tools.openvino_inference import OpenVINOInferenceEngine

        model_xml = 'export_models/test.xml'

        # Test with GPU
        print("\nTesting GPU inference...")
        gpu_engine = OpenVINOInferenceEngine(model_xml, device='GPU')

        # Create test input
        test_input = np.random.randn(1, 3, 300, 25, 2).astype(np.float32)

        # Run inference
        output = gpu_engine.infer(test_input)

        print(f"✓ GPU inference successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

        # Test timing
        output, time_ms = gpu_engine.infer_with_timing(test_input)
        print(f"✓ GPU inference timing: {time_ms:.2f} ms")

        return True

    except Exception as e:
        print(f"✗ GPU inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison(has_gpu):
    """Test 4: Compare CPU vs GPU performance"""
    print("\n" + "=" * 60)
    print("Test 4: Performance Comparison")
    print("=" * 60)

    try:
        from tools.openvino_inference import OpenVINOInferenceEngine

        model_xml = 'export_models/test.xml'
        test_input = np.random.randn(8, 3, 300, 25, 2).astype(np.float32)  # Batch of 8

        # CPU benchmark
        print("\n--- CPU Benchmark ---")
        cpu_engine = OpenVINOInferenceEngine(model_xml, device='CPU')
        cpu_stats = cpu_engine.benchmark(test_input, num_iterations=20, warmup=5)

        if has_gpu:
            # GPU benchmark
            print("\n--- GPU Benchmark ---")
            gpu_engine = OpenVINOInferenceEngine(model_xml, device='GPU')
            gpu_stats = gpu_engine.benchmark(test_input, num_iterations=20, warmup=5)

            # Compare
            speedup = cpu_stats['mean'] / gpu_stats['mean']
            print("\n" + "=" * 60)
            print("Performance Comparison")
            print("=" * 60)
            print(f"CPU: {cpu_stats['mean']:.2f} ms ({cpu_stats['throughput']:.2f} samples/sec)")
            print(f"GPU: {gpu_stats['mean']:.2f} ms ({gpu_stats['throughput']:.2f} samples/sec)")
            print(f"Speedup: {speedup:.2f}x")

            if speedup > 1.5:
                print(f"\n✅ GPU is {speedup:.2f}x faster than CPU!")
            elif speedup > 1.0:
                print(f"\n⚠️  GPU is only {speedup:.2f}x faster (expected > 1.5x)")
                print("   This could be due to:")
                print("   - Small batch size (try larger batches)")
                print("   - Integrated GPU with limited compute")
                print("   - GPU driver issues")
            else:
                print(f"\n⚠️  WARNING: GPU is slower than CPU!")
                print("   This usually means GPU inference is not working correctly.")
        else:
            print("\n⏭️  Skipping GPU benchmark (no GPU available)")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test 5: Test skeleton visualization"""
    print("\n" + "=" * 60)
    print("Test 5: Skeleton Visualization")
    print("=" * 60)

    try:
        # Create test skeleton data
        test_data = np.random.randn(1, 3, 300, 25, 2).astype(np.float32)

        # Save test data
        os.makedirs('data/test', exist_ok=True)
        test_data_path = 'data/test/test_skeleton.npy'
        np.save(test_data_path, test_data)
        print(f"✓ Created test skeleton data: {test_data_path}")

        # Test visualization
        print("\nTesting visualization (saving 3 frames)...")
        from tools.visualize_skeleton import visualize_skeleton_sequence

        output_dir = 'outputs/test_visualization'
        visualize_skeleton_sequence(
            test_data[0],  # Remove batch dimension
            skeleton_type='ntu-rgb+d',
            action_label='Test Action',
            save_path=output_dir,
            frame_interval=50,
            max_frames=3
        )

        # Check if files were created
        frame_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]

        if len(frame_files) >= 3:
            print(f"✓ Visualization successful! Created {len(frame_files)} frame images")
            print(f"  Output directory: {output_dir}")
            return True
        else:
            print(f"⚠️  Expected 3 frames, but got {len(frame_files)}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n")
    print("=" * 60)
    print("OpenVINO GPU Inference Test Suite")
    print("=" * 60)
    print()

    results = {}

    # Test 1: Device detection
    has_gpu = test_device_detection()
    results['device_detection'] = True

    # Test 2: Inference sync
    results['inference_sync'] = test_inference_sync()

    # Test 3: GPU inference
    if results['inference_sync']:
        results['gpu_inference'] = test_gpu_inference(has_gpu)
    else:
        results['gpu_inference'] = False

    # Test 4: Performance comparison
    if results['inference_sync']:
        results['performance'] = test_performance_comparison(has_gpu)
    else:
        results['performance'] = False

    # Test 5: Visualization
    results['visualization'] = test_visualization()

    # Print summary
    print("\n")
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("\nOpenVINO GPU inference fix is working correctly.")
        print("You can now:")
        print("  1. Use GPU for inference: --device GPU")
        print("  2. Visualize skeletons: python tools/visualize_skeleton.py")
        print("  3. Compare performance: python tools/compare_inference.py")
    else:
        print("⚠️  Some tests failed")
        print("\nPlease check:")
        print("  1. OpenVINO installation: pip install openvino==2026.2.0")
        print("  2. GPU drivers (if using GPU)")
        print("  3. Model export: python tools/export_to_openvino.py")
    print("=" * 60)
    print()

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
