#!/usr/bin/env python
"""
Test script for PyTorch vs OpenVINO performance comparison
"""
import os
import sys
import json

def test_accuracy_only():
    """Test 1: Accuracy comparison only"""
    print("=" * 60)
    print("Test 1: Accuracy Comparison Only")
    print("=" * 60)

    cmd = """python tools/compare_inference.py \
        --weights fake_models/test.pt \
        --openvino-xml export_models/test.xml \
        --accuracy-only \
        --num-samples 10 \
        --output-dir outputs/test_accuracy"""

    print(f"\nCommand: {cmd}\n")
    result = os.system(cmd)

    if result == 0:
        print("\n✓ Test 1 passed: Accuracy comparison completed")

        # Check result file
        result_file = "outputs/test_accuracy/comparison_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)

            accuracy = results.get('accuracy', {})
            output_comp = accuracy.get('output_comparison', {})
            pred_comp = accuracy.get('prediction_comparison', {})

            print(f"\n  Outputs match: {output_comp.get('is_close', 'N/A')}")
            print(f"  RMSE: {output_comp.get('rmse', 'N/A')}")
            print(f"  Top-1 match rate: {pred_comp.get('top1_match_rate', 'N/A'):.2%}")
            print(f"  Top-5 match rate: {pred_comp.get('topk_match_rate', 'N/A'):.2%}")

        return True
    else:
        print("\n✗ Test 1 failed")
        return False


def test_benchmark_only():
    """Test 2: Performance benchmark only"""
    print("\n" + "=" * 60)
    print("Test 2: Performance Benchmark Only")
    print("=" * 60)

    cmd = """python tools/compare_inference.py \
        --weights fake_models/test.pt \
        --openvino-xml export_models/test.xml \
        --device CPU \
        --benchmark-only \
        --num-iterations 30 \
        --benchmark-batch-sizes "1,4" \
        --output-dir outputs/test_benchmark"""

    print(f"\nCommand: {cmd}\n")
    result = os.system(cmd)

    if result == 0:
        print("\n✓ Test 2 passed: Performance benchmark completed")

        # Check result file
        result_file = "outputs/test_benchmark/comparison_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)

            performance = results.get('performance', {})

            for batch_key in ['batch_1', 'batch_4']:
                if batch_key in performance:
                    batch_result = performance[batch_key]
                    batch_size = batch_result['batch_size']
                    pytorch_mean = batch_result['pytorch']['mean']
                    openvino_mean = batch_result['openvino']['mean']
                    speedup = batch_result['speedup']

                    print(f"\n  Batch Size {batch_size}:")
                    print(f"    PyTorch:  {pytorch_mean:.2f} ms")
                    print(f"    OpenVINO: {openvino_mean:.2f} ms")
                    print(f"    Speedup:  {speedup:.2f}x")

        return True
    else:
        print("\n✗ Test 2 failed")
        return False


def test_full_comparison():
    """Test 3: Full comparison (accuracy + performance)"""
    print("\n" + "=" * 60)
    print("Test 3: Full Comparison (Accuracy + Performance)")
    print("=" * 60)

    cmd = """python tools/compare_inference.py \
        --weights fake_models/test.pt \
        --openvino-xml export_models/test.xml \
        --device CPU \
        --num-iterations 30 \
        --benchmark-batch-sizes "1,4" \
        --num-samples 10 \
        --output-dir outputs/test_full"""

    print(f"\nCommand: {cmd}\n")
    result = os.system(cmd)

    if result == 0:
        print("\n✓ Test 3 passed: Full comparison completed")

        # Check result file
        result_file = "outputs/test_full/comparison_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)

            # Check both accuracy and performance results
            has_accuracy = results.get('accuracy') is not None
            has_performance = results.get('performance') is not None

            print(f"\n  Has accuracy results: {has_accuracy}")
            print(f"  Has performance results: {has_performance}")

            if has_accuracy and has_performance:
                print("  ✓ Both accuracy and performance results present")
                return True
            else:
                print("  ✗ Missing accuracy or performance results")
                return False

        return True
    else:
        print("\n✗ Test 3 failed")
        return False


def test_multi_batch_sizes():
    """Test 4: Multiple batch sizes"""
    print("\n" + "=" * 60)
    print("Test 4: Multiple Batch Sizes")
    print("=" * 60)

    cmd = """python tools/compare_inference.py \
        --weights fake_models/test.pt \
        --openvino-xml export_models/test.xml \
        --device CPU \
        --benchmark-only \
        --num-iterations 20 \
        --benchmark-batch-sizes "1,2,4,8" \
        --output-dir outputs/test_multi_batch"""

    print(f"\nCommand: {cmd}\n")
    result = os.system(cmd)

    if result == 0:
        print("\n✓ Test 4 passed: Multiple batch sizes tested")

        # Check result file
        result_file = "outputs/test_multi_batch/comparison_results.json"
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)

            performance = results.get('performance', {})
            batch_keys = ['batch_1', 'batch_2', 'batch_4', 'batch_8']

            print(f"\n  Batch sizes tested: {len([k for k in batch_keys if k in performance])}/4")

            # Print summary
            print("\n  Batch Size  PyTorch(ms)  OpenVINO(ms)  Speedup")
            print("  " + "-" * 50)
            for batch_key in batch_keys:
                if batch_key in performance:
                    batch_result = performance[batch_key]
                    batch_size = batch_result['batch_size']
                    pytorch_mean = batch_result['pytorch']['mean']
                    openvino_mean = batch_result['openvino']['mean']
                    speedup = batch_result['speedup']

                    print(f"  {batch_size:^10}  {pytorch_mean:>11.2f}  {openvino_mean:>12.2f}  {speedup:>7.2f}x")

        return True
    else:
        print("\n✗ Test 4 failed")
        return False


def prepare_models():
    """Prepare test models if not exist"""
    print("=" * 60)
    print("Preparing Test Models")
    print("=" * 60)

    model_pt = "fake_models/test.pt"
    model_xml = "export_models/test.xml"

    need_create = not os.path.exists(model_pt)
    need_export = not os.path.exists(model_xml)

    if need_create:
        print("\nCreating fake PyTorch model...")
        result = os.system("python tools/create_fake_model.py --output fake_models/test.pt")
        if result != 0:
            print("✗ Failed to create fake model")
            return False
        print("✓ Fake model created")
    else:
        print(f"\n✓ PyTorch model already exists: {model_pt}")

    if need_export:
        print("\nExporting to OpenVINO...")
        result = os.system("python tools/export_to_openvino.py --weights fake_models/test.pt --output-dir export_models")
        if result != 0:
            print("✗ Failed to export model")
            return False
        print("✓ OpenVINO model exported")
    else:
        print(f"✓ OpenVINO model already exists: {model_xml}")

    return True


def main():
    """Run all tests"""
    print("\n")
    print("=" * 60)
    print("PyTorch vs OpenVINO Performance Comparison Test Suite")
    print("=" * 60)
    print()

    # Prepare models
    if not prepare_models():
        print("\n❌ Failed to prepare models")
        return 1

    # Run tests
    results = {}

    try:
        results['accuracy_only'] = test_accuracy_only()
        results['benchmark_only'] = test_benchmark_only()
        results['full_comparison'] = test_full_comparison()
        results['multi_batch_sizes'] = test_multi_batch_sizes()

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:25s}: {status}")

        all_passed = all(results.values())

        print("\n" + "=" * 60)
        if all_passed:
            print("✅ All tests passed!")
            print("\nPerformance comparison is working correctly.")
            print("\nYou can now:")
            print("  1. Compare accuracy: --accuracy-only")
            print("  2. Compare performance: --benchmark-only")
            print("  3. Full comparison: (no extra flags)")
            print("  4. Test multiple batch sizes: --benchmark-batch-sizes \"1,4,8,16\"")
        else:
            print("⚠️  Some tests failed")
            print("\nPlease check:")
            print("  1. Model files exist: fake_models/test.pt, export_models/test.xml")
            print("  2. Dependencies installed: pip install -r requirements_openvino.txt")
        print("=" * 60)
        print()

        return 0 if all_passed else 1

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
