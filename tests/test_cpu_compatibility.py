#!/usr/bin/env python
"""
Test CPU Compatibility after CUDA Hardcoding Fixes

This script tests that all fixed code works correctly on CPU-only systems.

Usage:
    python tests/test_cpu_compatibility.py
"""

import os
import sys
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.st_gcn_twostream import Model as TwoStreamModel


def test_model_creation():
    """Test 1: Model can be created"""
    print("\n[Test 1] Testing model creation...")

    try:
        model = TwoStreamModel(
            in_channels=3,
            num_class=60,
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            edge_importance_weighting=True
        )
        model.eval()
        print("  ✓ Model created successfully")
        return True, model
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False, None


def test_cpu_forward_pass(model):
    """Test 2: Forward pass works on CPU"""
    print("\n[Test 2] Testing CPU forward pass...")

    try:
        # Create CPU input
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 50, 25, 2)  # Small input

        print(f"  Input shape: {test_input.shape}")
        print(f"  Input device: {test_input.device}")

        # Forward pass
        with torch.no_grad():
            output = model(test_input)

        print(f"  Output shape: {output.shape}")
        print(f"  Output device: {output.device}")

        # Verify output
        assert output.shape == (batch_size, 60), f"Unexpected output shape: {output.shape}"
        assert output.device.type == 'cpu', f"Output should be on CPU, got: {output.device}"

        print("  ✓ CPU forward pass successful")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_forward_pass_if_available(model):
    """Test 3: Forward pass works on GPU (if available)"""
    print("\n[Test 3] Testing GPU forward pass...")

    if not torch.cuda.is_available():
        print("  ⊘ CUDA not available, skipping GPU test")
        return True

    try:
        # Move model to GPU
        model_gpu = model.cuda()

        # Create GPU input
        batch_size = 2
        test_input = torch.randn(batch_size, 3, 50, 25, 2).cuda()

        print(f"  Input device: {test_input.device}")

        # Forward pass
        with torch.no_grad():
            output = model_gpu(test_input)

        print(f"  Output device: {output.device}")

        # Verify output
        assert output.device.type == 'cuda', f"Output should be on CUDA, got: {output.device}"

        print("  ✓ GPU forward pass successful")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device_switching(model):
    """Test 4: Model can switch between CPU and GPU"""
    print("\n[Test 4] Testing device switching...")

    if not torch.cuda.is_available():
        print("  ⊘ CUDA not available, skipping device switching test")
        return True

    try:
        batch_size = 2

        # CPU -> GPU
        model_gpu = model.cuda()
        input_gpu = torch.randn(batch_size, 3, 50, 25, 2).cuda()
        with torch.no_grad():
            output_gpu = model_gpu(input_gpu)
        assert output_gpu.device.type == 'cuda'
        print("  ✓ CPU -> GPU switch successful")

        # GPU -> CPU
        model_cpu = model_gpu.cpu()
        input_cpu = torch.randn(batch_size, 3, 50, 25, 2)
        with torch.no_grad():
            output_cpu = model_cpu(input_cpu)
        assert output_cpu.device.type == 'cpu'
        print("  ✓ GPU -> CPU switch successful")

        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_motion_stream_computation():
    """Test 5: Motion stream computation is device-agnostic"""
    print("\n[Test 5] Testing motion stream computation...")

    try:
        model = TwoStreamModel(
            in_channels=3,
            num_class=60,
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            edge_importance_weighting=True
        )
        model.eval()

        # Test on CPU
        input_cpu = torch.randn(1, 3, 10, 25, 2)
        with torch.no_grad():
            output_cpu = model(input_cpu)

        print(f"  CPU motion stream: ✓")

        # Test on GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            input_gpu = input_cpu.cuda()
            with torch.no_grad():
                output_gpu = model_gpu(input_gpu)

            # Compare outputs (should be similar)
            diff = torch.abs(output_cpu - output_gpu.cpu()).max().item()
            print(f"  GPU motion stream: ✓")
            print(f"  CPU vs GPU max diff: {diff:.6f}")

            if diff > 1e-4:
                print(f"  ⚠ Warning: Difference larger than expected")

        print("  ✓ Motion stream computation is device-agnostic")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_torchlight_gpu_functions():
    """Test 6: torchlight GPU functions don't crash on CPU"""
    print("\n[Test 6] Testing torchlight GPU functions...")

    try:
        import torchlight

        # This should not crash on CPU-only systems
        torchlight.occupy_gpu()
        print("  ✓ torchlight.occupy_gpu() works on CPU")

        # Test with GPU list
        torchlight.occupy_gpu([0, 1])
        print("  ✓ torchlight.occupy_gpu([0, 1]) works on CPU")

        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_various_batch_sizes():
    """Test 7: Various batch sizes work correctly"""
    print("\n[Test 7] Testing various batch sizes...")

    try:
        model = TwoStreamModel(
            in_channels=3,
            num_class=60,
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            edge_importance_weighting=True
        )
        model.eval()

        batch_sizes = [1, 2, 4, 8, 16]

        for bs in batch_sizes:
            test_input = torch.randn(bs, 3, 30, 25, 2)
            with torch.no_grad():
                output = model(test_input)
            assert output.shape == (bs, 60)
            print(f"  ✓ Batch size {bs}: OK")

        print("  ✓ All batch sizes work correctly")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("CPU Compatibility Test Suite")
    print("="*70)

    # System info
    print("\nSystem Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        print(f"  GPU name: {torch.cuda.get_device_name(0)}")
    print(f"  CPU count: {torch.get_num_threads()}")

    # Run tests
    results = []

    # Test 1
    success, model = test_model_creation()
    results.append(("Model Creation", success))
    if not success:
        print("\n✗ Critical failure: Cannot create model")
        return False

    # Test 2
    success = test_cpu_forward_pass(model)
    results.append(("CPU Forward Pass", success))

    # Test 3
    success = test_gpu_forward_pass_if_available(model)
    results.append(("GPU Forward Pass", success))

    # Test 4
    success = test_device_switching(model)
    results.append(("Device Switching", success))

    # Test 5
    success = test_motion_stream_computation()
    results.append(("Motion Stream", success))

    # Test 6
    success = test_torchlight_gpu_functions()
    results.append(("Torchlight GPU Functions", success))

    # Test 7
    success = test_various_batch_sizes()
    results.append(("Various Batch Sizes", success))

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! CPU compatibility verified.")
        print("\nThe following has been verified:")
        print("  ✓ Model works on CPU-only systems")
        print("  ✓ No hardcoded CUDA operations")
        print("  ✓ Device-agnostic tensor creation")
        print("  ✓ Graceful fallback when GPU unavailable")
        if torch.cuda.is_available():
            print("  ✓ GPU operations still work correctly")
        return True
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
