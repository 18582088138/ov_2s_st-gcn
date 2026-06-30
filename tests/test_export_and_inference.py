#!/usr/bin/env python
"""
Test script for model export and inference
"""
import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.st_gcn_twostream import Model as TwoStreamModel
from tools.export_to_openvino import export_to_onnx, convert_onnx_to_openvino
from tools.openvino_inference import OpenVINOInferenceEngine, TorchInferenceEngine


def test_model_creation():
    """Test creating 2S-STGCN model"""
    print("\n=== Test 1: Model Creation ===")

    try:
        model = TwoStreamModel(
            in_channels=3,
            num_class=60,
            graph_args={'layout': 'ntu-rgb+d', 'strategy': 'spatial'},
            edge_importance_weighting=True
        )
        model.eval()
        print("✓ Model created successfully")
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None


def test_torch_inference(model):
    """Test PyTorch inference"""
    print("\n=== Test 2: PyTorch Inference ===")

    try:
        # Create dummy input
        dummy_input = torch.randn(2, 3, 300, 25, 2)

        # Run inference
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ PyTorch inference successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")

        return True
    except Exception as e:
        print(f"✗ PyTorch inference failed: {e}")
        return False


def test_onnx_export(model, output_dir='./export_models'):
    """Test ONNX export"""
    print("\n=== Test 3: ONNX Export ===")

    try:
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, 'test_model.onnx')

        export_to_onnx(model, onnx_path, input_shape=(1, 3, 300, 25, 2))

        if os.path.exists(onnx_path):
            file_size = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"✓ ONNX export successful")
            print(f"  File: {onnx_path}")
            print(f"  Size: {file_size:.2f} MB")
            return onnx_path
        else:
            print("✗ ONNX file not created")
            return None
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        return None


def test_openvino_conversion(onnx_path, output_dir='./export_models'):
    """Test OpenVINO conversion"""
    print("\n=== Test 4: OpenVINO Conversion ===")

    try:
        xml_path, bin_path = convert_onnx_to_openvino(onnx_path, output_dir)

        if os.path.exists(xml_path) and os.path.exists(bin_path):
            xml_size = os.path.getsize(xml_path) / 1024
            bin_size = os.path.getsize(bin_path) / (1024 * 1024)
            print(f"✓ OpenVINO conversion successful")
            print(f"  XML: {xml_path} ({xml_size:.2f} KB)")
            print(f"  BIN: {bin_path} ({bin_size:.2f} MB)")
            return xml_path
        else:
            print("✗ OpenVINO files not created")
            return None
    except Exception as e:
        print(f"✗ OpenVINO conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_openvino_inference(xml_path):
    """Test OpenVINO inference"""
    print("\n=== Test 5: OpenVINO Inference ===")

    try:
        # Initialize engine
        engine = OpenVINOInferenceEngine(xml_path, device='CPU')

        # Create test input
        test_input = np.random.randn(2, 3, 300, 25, 2).astype(np.float32)

        # Run inference
        output = engine.infer(test_input)

        print(f"✓ OpenVINO inference successful")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")

        return engine, test_input, output
    except Exception as e:
        print(f"✗ OpenVINO inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_output_consistency(model, engine, test_input):
    """Test output consistency between PyTorch and OpenVINO"""
    print("\n=== Test 6: Output Consistency ===")

    try:
        # PyTorch inference
        torch_input = torch.from_numpy(test_input).float()
        with torch.no_grad():
            torch_output = model(torch_input).numpy()

        # OpenVINO inference
        openvino_output = engine.infer(test_input)

        # Compare outputs
        abs_diff = np.abs(torch_output - openvino_output)
        rel_diff = abs_diff / (np.abs(torch_output) + 1e-8)

        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)
        max_rel_diff = np.max(rel_diff)

        # Check if close enough
        is_close = np.allclose(torch_output, openvino_output, rtol=1e-3, atol=1e-3)

        print(f"  Max absolute difference: {max_abs_diff:.6f}")
        print(f"  Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"  Max relative difference: {max_rel_diff:.6f}")

        if is_close:
            print("✓ Outputs are consistent (within tolerance)")
            return True
        else:
            print("⚠ Outputs have larger differences than expected")
            print("  This might be due to numerical precision differences")
            return False

    except Exception as e:
        print(f"✗ Consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_comparison(model, engine, test_input, num_iterations=50):
    """Test performance comparison"""
    print("\n=== Test 7: Performance Comparison ===")

    try:
        # PyTorch performance
        torch_engine = TorchInferenceEngine(model)
        torch_stats = torch_engine.benchmark(test_input, num_iterations=num_iterations, warmup=5)

        # OpenVINO performance
        openvino_stats = engine.benchmark(test_input, num_iterations=num_iterations, warmup=5)

        # Calculate speedup
        speedup = torch_stats['mean'] / openvino_stats['mean']

        print(f"\n  PyTorch:  {torch_stats['mean']:.2f} ms")
        print(f"  OpenVINO: {openvino_stats['mean']:.2f} ms")
        print(f"  Speedup:  {speedup:.2f}x")

        if speedup > 1.0:
            print(f"✓ OpenVINO is {speedup:.2f}x faster than PyTorch")
        else:
            print(f"⚠ OpenVINO is slower ({speedup:.2f}x)")

        return True
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running 2S-STGCN Export and Inference Tests")
    print("="*60)

    results = {}

    # Test 1: Model creation
    model = test_model_creation()
    results['model_creation'] = model is not None

    if model is None:
        print("\n✗ Cannot proceed without model")
        return results

    # Test 2: PyTorch inference
    results['torch_inference'] = test_torch_inference(model)

    # Test 3: ONNX export
    onnx_path = test_onnx_export(model)
    results['onnx_export'] = onnx_path is not None

    if onnx_path is None:
        print("\n✗ Cannot proceed without ONNX model")
        return results

    # Test 4: OpenVINO conversion
    xml_path = test_openvino_conversion(onnx_path)
    results['openvino_conversion'] = xml_path is not None

    if xml_path is None:
        print("\n⚠ OpenVINO not installed or conversion failed")
        print("  Install OpenVINO: pip install openvino openvino-dev")
        return results

    # Test 5: OpenVINO inference
    engine, test_input, output = test_openvino_inference(xml_path)
    results['openvino_inference'] = engine is not None

    if engine is None:
        return results

    # Test 6: Output consistency
    results['output_consistency'] = test_output_consistency(model, engine, test_input)

    # Test 7: Performance comparison
    results['performance_comparison'] = test_performance_comparison(model, engine, test_input)

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")

    return results


if __name__ == '__main__':
    run_all_tests()
