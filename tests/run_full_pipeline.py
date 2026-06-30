#!/usr/bin/env python
"""
Complete pipeline test with real pretrained model
Demonstrates the full workflow from model export to inference comparison
"""
import os
import sys
import argparse
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def step1_export_model(weights_path, output_dir, num_class=60, layout='ntu-rgb+d'):
    """Step 1: Export PyTorch model to OpenVINO"""
    print("\n" + "="*70)
    print("STEP 1: Export PyTorch Model to OpenVINO IR")
    print("="*70)

    from tools.export_to_openvino import load_pretrained_model, export_to_onnx, convert_onnx_to_openvino

    # Load model
    print("\n[1/3] Loading PyTorch model...")
    model = load_pretrained_model(weights_path, num_class, layout)

    # Export to ONNX
    print("\n[2/3] Exporting to ONNX...")
    model_name = os.path.splitext(os.path.basename(weights_path))[0]
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    export_to_onnx(model, onnx_path, input_shape=(1, 3, 300, 25, 2))

    # Convert to OpenVINO
    print("\n[3/3] Converting to OpenVINO IR...")
    try:
        xml_path, bin_path = convert_onnx_to_openvino(onnx_path, output_dir)
        print(f"\n✓ Export complete!")
        print(f"  ONNX: {onnx_path}")
        print(f"  XML:  {xml_path}")
        print(f"  BIN:  {bin_path}")
        return xml_path, model
    except Exception as e:
        print(f"\n✗ OpenVINO conversion failed: {e}")
        print("\nPlease install OpenVINO:")
        print("  pip install openvino openvino-dev")
        return None, model


def step2_compare_inference(weights_path, xml_path, data_path, num_samples, num_iterations, output_dir):
    """Step 2: Compare PyTorch and OpenVINO inference"""
    print("\n" + "="*70)
    print("STEP 2: Compare Inference Performance and Accuracy")
    print("="*70)

    import subprocess

    cmd = [
        sys.executable,
        'tools/compare_inference.py',
        '--weights', weights_path,
        '--openvino-xml', xml_path,
        '--num-samples', str(num_samples),
        '--num-iterations', str(num_iterations),
        '--output-dir', output_dir
    ]

    if data_path:
        cmd.extend(['--data-path', data_path])

    print(f"\nRunning comparison...")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if result.returncode == 0:
        print("\n✓ Comparison complete!")
        result_file = os.path.join(output_dir, 'comparison_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results = json.load(f)
            return results
    else:
        print("\n✗ Comparison failed!")

    return None


def step3_test_e2e_pipeline(xml_path, data_path, label_path, output_dir):
    """Step 3: Test OpenVINO E2E pipeline"""
    print("\n" + "="*70)
    print("STEP 3: Test OpenVINO End-to-End Pipeline")
    print("="*70)

    import subprocess

    if data_path and label_path and os.path.exists(data_path) and os.path.exists(label_path):
        # Run evaluation mode
        print("\n[Running evaluation mode with real data]")
        cmd = [
            sys.executable,
            'tools/openvino_e2e_pipeline.py',
            '--model-xml', xml_path,
            '--mode', 'evaluate',
            '--data-path', data_path,
            '--label-path', label_path,
            '--batch-size', '1',
            '--max-samples', '100',
            '--output-dir', output_dir,
            '--device', 'GPU'
        ]
    else:
        # Run prediction mode
        print("\n[Running prediction mode with sample data]")
        cmd = [
            sys.executable,
            'tools/openvino_e2e_pipeline.py',
            '--model-xml', xml_path,
            '--mode', 'predict',
            '--output-dir', output_dir,
            '--device', 'GPU'
        ]

        if data_path:
            cmd.extend(['--data-path', data_path])

    print(f"\nRunning E2E pipeline...")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if result.returncode == 0:
        print("\n✓ E2E pipeline test complete!")
        return True
    else:
        print("\n✗ E2E pipeline test failed!")
        return False


def generate_summary_report(export_results, comparison_results, output_dir):
    """Generate final summary report"""
    print("\n" + "="*70)
    print("FINAL SUMMARY REPORT")
    print("="*70)

    report = {
        'export_status': 'success' if export_results else 'failed',
        'comparison_results': comparison_results
    }

    print("\n1. Model Export:")
    if export_results:
        print("   ✓ Successfully exported PyTorch model to OpenVINO IR")
    else:
        print("   ✗ Export failed (OpenVINO not installed?)")

    if comparison_results:
        print("\n2. Accuracy Comparison:")
        acc = comparison_results.get('accuracy', {})
        output_comp = acc.get('output_comparison', {})
        pred_comp = acc.get('prediction_comparison', {})

        print(f"   - Outputs match: {output_comp.get('is_close', 'N/A')}")
        print(f"   - Max absolute diff: {output_comp.get('max_abs_diff', 'N/A'):.6f}")
        print(f"   - Mean absolute diff: {output_comp.get('mean_abs_diff', 'N/A'):.6f}")
        print(f"   - Top-1 match rate: {pred_comp.get('top1_match_rate', 'N/A'):.2%}")

        print("\n3. Performance Comparison:")
        perf = comparison_results.get('performance', {})
        pytorch_perf = perf.get('pytorch', {})
        openvino_perf = perf.get('openvino', {})
        speedup = perf.get('speedup', 0)

        print(f"   - PyTorch:  {pytorch_perf.get('mean', 'N/A'):.2f} ms ({pytorch_perf.get('fps', 'N/A'):.2f} FPS)")
        print(f"   - OpenVINO: {openvino_perf.get('mean', 'N/A'):.2f} ms ({openvino_perf.get('fps', 'N/A'):.2f} FPS)")
        print(f"   - Speedup:  {speedup:.2f}x")

        if speedup > 1.0:
            print(f"\n   🚀 OpenVINO is {speedup:.2f}x faster!")
        else:
            print(f"\n   ⚠ Performance needs investigation")

    # Save report
    report_file = os.path.join(output_dir, 'final_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Full report saved to: {report_file}")
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Complete pipeline: Export and test 2S-STGCN with OpenVINO'
    )
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to PyTorch weights (.pt file)')
    parser.add_argument('--data-path', type=str, default='',
                        help='Path to test data (.npy file)')
    parser.add_argument('--label-path', type=str, default='',
                        help='Path to test labels (.pkl file)')
    parser.add_argument('--num-class', type=int, default=60,
                        help='Number of action classes')
    parser.add_argument('--layout', type=str, default='ntu-rgb+d',
                        help='Skeleton layout')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for comparison')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of iterations for benchmarking')
    parser.add_argument('--export-dir', type=str, default='./export_models',
                        help='Directory for exported models')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Directory for output results')
    parser.add_argument('--skip-export', action='store_true',
                        help='Skip export step (use existing model)')
    parser.add_argument('--xml-path', type=str, default='',
                        help='Path to existing OpenVINO XML (if --skip-export)')

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.export_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*70)
    print("2S-STGCN OpenVINO Complete Pipeline Test")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Weights:     {args.weights}")
    print(f"  Data:        {args.data_path if args.data_path else 'Random samples'}")
    print(f"  Export dir:  {args.export_dir}")
    print(f"  Output dir:  {args.output_dir}")

    # Step 1: Export model
    if not args.skip_export:
        xml_path, model = step1_export_model(
            args.weights,
            args.export_dir,
            args.num_class,
            args.layout
        )
        export_success = xml_path is not None
    else:
        if not args.xml_path:
            print("\nERROR: --xml-path required when using --skip-export")
            return
        xml_path = args.xml_path
        export_success = os.path.exists(xml_path)
        print(f"\nUsing existing model: {xml_path}")

    if not export_success:
        print("\n✗ Pipeline stopped: Cannot proceed without OpenVINO model")
        return

    # Step 2: Compare inference
    comparison_results = step2_compare_inference(
        args.weights,
        xml_path,
        args.data_path,
        args.num_samples,
        args.num_iterations,
        args.output_dir
    )

    # Step 3: Test E2E pipeline
    step3_test_e2e_pipeline(
        xml_path,
        args.data_path,
        args.label_path,
        args.output_dir
    )

    # Generate summary
    generate_summary_report(export_success, comparison_results, args.output_dir)

    print("\n✅ Pipeline complete! Check the outputs directory for results.")


if __name__ == '__main__':
    main()
