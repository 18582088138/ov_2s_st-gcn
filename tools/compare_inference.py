#!/usr/bin/env python
"""
Compare inference performance and accuracy between PyTorch and OpenVINO
"""
import os
import sys
import argparse
import json
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.openvino_inference import OpenVINOInferenceEngine, TorchInferenceEngine
from tools.export_to_openvino import load_pretrained_model


def compare_outputs(torch_output, openvino_output, rtol=1e-3, atol=1e-3):
    """
    Compare outputs from PyTorch and OpenVINO

    Args:
        torch_output: PyTorch model output
        openvino_output: OpenVINO model output
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        dict: Comparison metrics
    """
    # Calculate differences
    abs_diff = np.abs(torch_output - openvino_output)
    rel_diff = abs_diff / (np.abs(torch_output) + 1e-8)

    # Check if outputs match within tolerance
    is_close = np.allclose(torch_output, openvino_output, rtol=rtol, atol=atol)

    metrics = {
        'is_close': is_close,
        'max_abs_diff': np.max(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'max_rel_diff': np.max(rel_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'rmse': np.sqrt(np.mean((torch_output - openvino_output) ** 2))
    }

    return metrics


def compare_predictions(torch_output, openvino_output, top_k=5):
    """
    Compare top-k predictions

    Args:
        torch_output: PyTorch predictions (N, num_classes)
        openvino_output: OpenVINO predictions (N, num_classes)
        top_k: Number of top predictions to compare

    Returns:
        dict: Prediction comparison
    """
    batch_size = torch_output.shape[0]

    # Get top-k predictions
    torch_topk = np.argsort(torch_output, axis=1)[:, -top_k:][:, ::-1]
    openvino_topk = np.argsort(openvino_output, axis=1)[:, -top_k:][:, ::-1]

    # Calculate top-1 and top-k accuracy
    top1_match = (torch_topk[:, 0] == openvino_topk[:, 0]).sum()
    topk_match = sum([len(set(t) & set(o)) for t, o in zip(torch_topk, openvino_topk)])

    results = {
        'top1_match_rate': top1_match / batch_size,
        'topk_match_rate': topk_match / (batch_size * top_k),
        'torch_top1': torch_topk[:, 0].tolist(),
        'openvino_top1': openvino_topk[:, 0].tolist()
    }

    return results


def load_sample_data(data_path, num_samples=10):
    """
    Load sample data for testing

    Args:
        data_path: Path to .npy data file
        num_samples: Number of samples to load

    Returns:
        numpy array with shape (N, C, T, V, M)
    """
    if os.path.exists(data_path):
        print(f"Loading data from: {data_path}")
        data = np.load(data_path, mmap_mode='r')

        # Limit number of samples
        if num_samples > 0:
            data = data[:num_samples]

        print(f"Loaded data shape: {data.shape}")
        return data
    else:
        print(f"Data file not found: {data_path}")
        print("Generating random sample data...")
        # Generate random data (N, C, T, V, M)
        data = np.random.randn(num_samples, 3, 300, 25, 2).astype(np.float32)
        return data


def save_results(results, output_path):
    """
    Save comparison results to JSON

    Args:
        results: Results dictionary
        output_path: Output JSON file path
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    results = convert_types(results)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare PyTorch and OpenVINO inference'
    )
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to PyTorch weights (.pt file)')
    parser.add_argument('--openvino-xml', type=str, required=True,
                        help='Path to OpenVINO IR XML file')
    parser.add_argument('--data-path', type=str, default='',
                        help='Path to test data (.npy file)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of iterations for benchmarking')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--benchmark-batch-sizes', type=str, default='1,4,8',
                        help='Comma-separated batch sizes to benchmark (e.g., "1,4,8,16")')
    parser.add_argument('--accuracy-only', action='store_true',
                        help='Only compare accuracy, skip performance benchmark')
    parser.add_argument('--benchmark-only', action='store_true',
                        help='Only run performance benchmark, skip accuracy comparison')
    parser.add_argument('--num-class', type=int, default=60,
                        help='Number of action classes')
    parser.add_argument('--layout', type=str, default='ntu-rgb+d',
                        help='Skeleton layout')
    parser.add_argument('--device', type=str, default='CPU',
                        help='OpenVINO device (CPU, GPU, etc.)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load sample data
    print("\n=== Step 1: Loading Data ===")
    if args.data_path and os.path.exists(args.data_path):
        test_data = load_sample_data(args.data_path, args.num_samples)
    else:
        print("Generating random test data...")
        test_data = np.random.randn(args.num_samples, 3, 300, 25, 2).astype(np.float32)
        print(f"Generated data shape: {test_data.shape}")

    # Load PyTorch model
    print("\n=== Step 2: Loading PyTorch Model ===")
    torch_model = load_pretrained_model(args.weights, args.num_class, args.layout)
    torch_engine = TorchInferenceEngine(torch_model)

    # Load OpenVINO model
    print("\n=== Step 3: Loading OpenVINO Model ===")
    openvino_engine = OpenVINOInferenceEngine(args.openvino_xml, args.device)

    # Step 4: Accuracy Comparison
    output_metrics = None
    prediction_metrics = None

    if not args.benchmark_only:
        print("\n=== Step 4: Comparing Outputs (Accuracy) ===")
        torch_outputs = []
        openvino_outputs = []

        for i, sample in enumerate(test_data):
            sample_batch = np.expand_dims(sample, axis=0)  # Add batch dimension

            torch_output = torch_engine.infer(sample_batch)
            openvino_output = openvino_engine.infer(sample_batch)

            torch_outputs.append(torch_output)
            openvino_outputs.append(openvino_output)

            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(test_data)} samples")

        torch_outputs = np.concatenate(torch_outputs, axis=0)
        openvino_outputs = np.concatenate(openvino_outputs, axis=0)

        # Calculate accuracy metrics
        output_metrics = compare_outputs(torch_outputs, openvino_outputs)
        prediction_metrics = compare_predictions(torch_outputs, openvino_outputs)

        print("\n=== Output Comparison ===")
        print(f"Outputs match (within tolerance): {output_metrics['is_close']}")
        print(f"Max absolute difference: {output_metrics['max_abs_diff']:.6f}")
        print(f"Mean absolute difference: {output_metrics['mean_abs_diff']:.6f}")
        print(f"Max relative difference: {output_metrics['max_rel_diff']:.6f}")
        print(f"Mean relative difference: {output_metrics['mean_rel_diff']:.6f}")
        print(f"RMSE: {output_metrics['rmse']:.6f}")

        print("\n=== Prediction Comparison ===")
        print(f"Top-1 match rate: {prediction_metrics['top1_match_rate']:.2%}")
        print(f"Top-5 match rate: {prediction_metrics['topk_match_rate']:.2%}")
    else:
        print("\n⏭️  Skipping accuracy comparison (--benchmark-only mode)")

    # Step 5: Performance Benchmarking
    performance_results = {}

    if not args.accuracy_only:
        print("\n=== Step 5: Performance Benchmarking ===")

        # Parse batch sizes
        batch_sizes = [int(b) for b in args.benchmark_batch_sizes.split(',')]
        print(f"Benchmarking batch sizes: {batch_sizes}")
        print(f"Iterations per batch size: {args.num_iterations}")
        print(f"Warmup iterations: {args.warmup}\n")

        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"Batch Size: {batch_size}")
            print(f"{'='*60}")

            # Create benchmark batch
            if batch_size == 1:
                benchmark_sample = np.expand_dims(test_data[0], axis=0)
            else:
                # Replicate first sample to create batch
                benchmark_sample = np.repeat(test_data[0:1], batch_size, axis=0)

            print(f"Benchmark input shape: {benchmark_sample.shape}")

            # PyTorch benchmark
            print(f"\n--- PyTorch Benchmark (batch_size={batch_size}) ---")
            torch_perf = torch_engine.benchmark(
                benchmark_sample,
                num_iterations=args.num_iterations,
                warmup=args.warmup
            )

            # OpenVINO benchmark
            print(f"\n--- OpenVINO Benchmark (batch_size={batch_size}, device={args.device}) ---")
            openvino_perf = openvino_engine.benchmark(
                benchmark_sample,
                num_iterations=args.num_iterations,
                warmup=args.warmup
            )

            # Calculate speedup
            speedup = torch_perf['mean'] / openvino_perf['mean']
            throughput_speedup = openvino_perf['throughput'] / torch_perf['throughput']

            # Store results
            performance_results[f'batch_{batch_size}'] = {
                'batch_size': batch_size,
                'pytorch': torch_perf,
                'openvino': openvino_perf,
                'speedup': float(speedup),
                'throughput_speedup': float(throughput_speedup)
            }

            # Print comparison
            print(f"\n{'='*60}")
            print(f"Performance Comparison (Batch Size: {batch_size})")
            print(f"{'='*60}")
            print(f"{'Metric':<25} {'PyTorch':>15} {'OpenVINO':>15} {'Speedup':>10}")
            print(f"{'-'*70}")
            print(f"{'Mean Latency (ms)':<25} {torch_perf['mean']:>15.2f} {openvino_perf['mean']:>15.2f} {speedup:>9.2f}x")
            print(f"{'Median Latency (ms)':<25} {torch_perf['median']:>15.2f} {openvino_perf['median']:>15.2f} {torch_perf['median']/openvino_perf['median']:>9.2f}x")
            print(f"{'P95 Latency (ms)':<25} {torch_perf['p95']:>15.2f} {openvino_perf['p95']:>15.2f} {torch_perf['p95']/openvino_perf['p95']:>9.2f}x")
            print(f"{'Min Latency (ms)':<25} {torch_perf['min']:>15.2f} {openvino_perf['min']:>15.2f} {torch_perf['min']/openvino_perf['min']:>9.2f}x")
            print(f"{'Max Latency (ms)':<25} {torch_perf['max']:>15.2f} {openvino_perf['max']:>15.2f} {torch_perf['max']/openvino_perf['max']:>9.2f}x")
            print(f"{'-'*70}")
            print(f"{'Throughput (samples/s)':<25} {torch_perf['throughput']:>15.2f} {openvino_perf['throughput']:>15.2f} {throughput_speedup:>9.2f}x")
            print(f"{'Per-sample Latency (ms)':<25} {torch_perf['mean']/batch_size:>15.2f} {openvino_perf['mean']/batch_size:>15.2f}")

        # Summary across all batch sizes
        print(f"\n{'='*60}")
        print("Performance Summary (All Batch Sizes)")
        print(f"{'='*60}")
        print(f"{'Batch Size':<15} {'PyTorch (ms)':>15} {'OpenVINO (ms)':>15} {'Speedup':>10}")
        print(f"{'-'*60}")
        for batch_size in batch_sizes:
            result = performance_results[f'batch_{batch_size}']
            print(f"{batch_size:<15} {result['pytorch']['mean']:>15.2f} {result['openvino']['mean']:>15.2f} {result['speedup']:>9.2f}x")
        print(f"{'='*60}")

    else:
        print("\n⏭️  Skipping performance benchmark (--accuracy-only mode)")

    # Compile results
    results = {
        'configuration': {
            'pytorch_weights': args.weights,
            'openvino_model': args.openvino_xml,
            'num_samples': args.num_samples,
            'num_iterations': args.num_iterations,
            'warmup': args.warmup,
            'benchmark_batch_sizes': batch_sizes if not args.accuracy_only else None,
            'device': args.device,
            'data_shape': list(test_data.shape),
            'accuracy_only': args.accuracy_only,
            'benchmark_only': args.benchmark_only
        },
        'accuracy': {
            'output_comparison': output_metrics,
            'prediction_comparison': prediction_metrics
        } if not args.benchmark_only else None,
        'performance': performance_results if not args.accuracy_only else None
    }

    # Save results
    output_file = os.path.join(args.output_dir, 'comparison_results.json')
    save_results(results, output_file)

    print("\n=== Comparison Complete! ===")


if __name__ == '__main__':
    main()
