#!/usr/bin/env python
"""
End-to-End OpenVINO Inference Pipeline for 2S-STGCN
Equivalent to PyTorch inference pipeline
"""
import os
import sys
import argparse
import json
import time
import pickle
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.openvino_inference import OpenVINOInferenceEngine


class OpenVINOActionRecognitionPipeline:
    """
    Complete end-to-end pipeline for skeleton-based action recognition using OpenVINO
    """

    def __init__(self, model_xml, label_map=None, device='CPU'):
        """
        Initialize the pipeline

        Args:
            model_xml: Path to OpenVINO IR XML file
            label_map: Dictionary mapping class indices to action names
            device: Target device (CPU, GPU, etc.)
        """
        self.engine = OpenVINOInferenceEngine(model_xml, device)
        self.label_map = label_map or {}
        self.device = device

        print(f"OpenVINO Action Recognition Pipeline initialized on {device}")

    def preprocess(self, skeleton_data):
        """
        Preprocess skeleton data

        Args:
            skeleton_data: Raw skeleton data with shape (C, T, V, M) or (N, C, T, V, M)

        Returns:
            Preprocessed data ready for inference
        """
        # Ensure 5D tensor (N, C, T, V, M)
        if skeleton_data.ndim == 4:
            skeleton_data = np.expand_dims(skeleton_data, axis=0)

        # Ensure float32
        if skeleton_data.dtype != np.float32:
            skeleton_data = skeleton_data.astype(np.float32)

        return skeleton_data

    def predict(self, skeleton_data, top_k=5):
        """
        Predict action from skeleton data

        Args:
            skeleton_data: Input skeleton data
            top_k: Number of top predictions to return

        Returns:
            dict: Prediction results
        """
        # Preprocess
        input_data = self.preprocess(skeleton_data)

        # Inference
        start_time = time.perf_counter()
        output = self.engine.infer(input_data)
        inference_time = (time.perf_counter() - start_time) * 1000

        # Get predictions
        predictions = self._process_output(output, top_k)
        predictions['inference_time_ms'] = inference_time

        return predictions

    def predict_batch(self, skeleton_batch):
        """
        Predict actions for a batch of skeleton sequences

        Args:
            skeleton_batch: Batch of skeleton data (N, C, T, V, M)

        Returns:
            list: List of prediction results for each sample
        """
        # Preprocess
        input_data = self.preprocess(skeleton_batch)

        # Inference
        start_time = time.perf_counter()
        output = self.engine.infer(input_data)
        inference_time = (time.perf_counter() - start_time) * 1000

        # Process each sample in batch
        results = []
        for i in range(output.shape[0]):
            pred = self._process_output(output[i:i+1], top_k=5)
            results.append(pred)

        # Add timing info
        avg_time = inference_time / output.shape[0]
        for result in results:
            result['inference_time_ms'] = avg_time

        return results

    def _process_output(self, output, top_k=5):
        """
        Process model output to get predictions

        Args:
            output: Model output logits (1, num_classes) or (num_classes,)
            top_k: Number of top predictions

        Returns:
            dict: Processed predictions
        """
        if output.ndim == 2:
            output = output[0]

        # Get top-k predictions
        top_indices = np.argsort(output)[-top_k:][::-1]
        top_scores = output[top_indices]

        # Apply softmax
        exp_scores = np.exp(top_scores - np.max(top_scores))
        probabilities = exp_scores / np.sum(exp_scores)

        # Format results
        predictions = []
        for idx, prob in zip(top_indices, probabilities):
            action_name = self.label_map.get(int(idx), f"Class_{idx}")
            predictions.append({
                'class_id': int(idx),
                'action': action_name,
                'probability': float(prob),
                'score': float(output[idx])
            })

        return {
            'top_prediction': predictions[0],
            'top_k_predictions': predictions
        }

    def benchmark(self, data_path, num_iterations=100, warmup=10, max_samples=None):
        """
        Benchmark inference latency over multiple iterations

        Args:
            data_path: Path to test data (.npy file) or None to use random data
            num_iterations: Number of inference iterations
            warmup: Number of warmup iterations
            max_samples: Number of samples to benchmark (None = use all)

        Returns:
            dict: Benchmark results
        """
        print(f"\n=== Benchmarking Inference Latency ===")

        # Load or generate data
        if data_path and os.path.exists(data_path):
            print(f"Loading data from: {data_path}")
            data = np.load(data_path, mmap_mode='r')

            # Apply max_samples limit
            if max_samples and max_samples < len(data):
                print(f"Using {max_samples} samples from dataset")
                data = data[:max_samples]
            else:
                max_samples = len(data)
                print(f"Using all {max_samples} samples from dataset")
        else:
            if max_samples is None:
                max_samples = 10
            print(f"Generating {max_samples} random samples...")
            data = np.random.randn(max_samples, 3, 300, 25, 2).astype(np.float32)

        num_samples = len(data)
        print(f"\nBenchmark configuration:")
        print(f"  Device: {self.device}")
        print(f"  Samples: {num_samples}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Warmup: {warmup}")

        # Warmup phase
        print(f"\nWarmup phase ({warmup} iterations)...")
        for i in range(warmup):
            sample_idx = i % num_samples
            sample = self.preprocess(data[sample_idx])
            _ = self.engine.infer(sample)

        # Additional GPU warmup if needed
        if 'GPU' in self.device.upper():
            gpu_warmup = 20
            print(f"Additional GPU warmup ({gpu_warmup} iterations)...")
            for i in range(gpu_warmup):
                sample_idx = i % num_samples
                sample = self.preprocess(data[sample_idx])
                _ = self.engine.infer(sample)

        # Benchmark phase
        print(f"\nRunning benchmark ({num_iterations} iterations)...")
        latencies = []

        for i in range(num_iterations):
            # Cycle through samples
            sample_idx = i % num_samples
            sample = self.preprocess(data[sample_idx])

            # Measure single inference latency
            start_time = time.perf_counter()
            _ = self.engine.infer(sample)
            end_time = time.perf_counter()

            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)

            if (i + 1) % 20 == 0:
                avg_so_far = np.mean(latencies)
                print(f"  Progress: {i + 1}/{num_iterations} - Avg latency: {avg_so_far:.2f} ms")

        # Calculate statistics
        latencies = np.array(latencies)

        results = {
            'device': self.device,
            'num_samples': num_samples,
            'num_iterations': num_iterations,
            'latency_stats': {
                'mean_ms': float(np.mean(latencies)),
                'std_ms': float(np.std(latencies)),
                'min_ms': float(np.min(latencies)),
                'max_ms': float(np.max(latencies)),
                'median_ms': float(np.median(latencies)),
                'p50_ms': float(np.percentile(latencies, 50)),
                'p90_ms': float(np.percentile(latencies, 90)),
                'p95_ms': float(np.percentile(latencies, 95)),
                'p99_ms': float(np.percentile(latencies, 99)),
            },
            'throughput': {
                'fps': float(1000.0 / np.mean(latencies)),
                'samples_per_sec': float(1000.0 / np.mean(latencies))
            }
        }

        # Print results
        print("\n" + "=" * 60)
        print("Benchmark Results")
        print("=" * 60)
        print(f"Device: {results['device']}")
        print(f"Samples tested: {results['num_samples']}")
        print(f"Iterations: {results['num_iterations']}")
        print(f"\n--- Latency Statistics ---")
        print(f"Mean:   {results['latency_stats']['mean_ms']:.2f} ms")
        print(f"Std:    {results['latency_stats']['std_ms']:.2f} ms")
        print(f"Min:    {results['latency_stats']['min_ms']:.2f} ms")
        print(f"Max:    {results['latency_stats']['max_ms']:.2f} ms")
        print(f"Median: {results['latency_stats']['median_ms']:.2f} ms")
        print(f"P90:    {results['latency_stats']['p90_ms']:.2f} ms")
        print(f"P95:    {results['latency_stats']['p95_ms']:.2f} ms")
        print(f"P99:    {results['latency_stats']['p99_ms']:.2f} ms")
        print(f"\n--- Throughput ---")
        print(f"FPS: {results['throughput']['fps']:.2f}")
        print("=" * 60)

        return results

    def evaluate(self, data_path, label_path, batch_size=1, max_samples=None):
        """
        Evaluate model on test dataset

        Args:
            data_path: Path to test data (.npy file)
            label_path: Path to test labels (.pkl file)
            batch_size: Batch size for inference
            max_samples: Maximum number of samples to evaluate

        Returns:
            dict: Evaluation results
        """
        print(f"\n=== Evaluating on {data_path} ===")

        # Load data
        print("Loading data...")
        data = np.load(data_path, mmap_mode='r')
        with open(label_path, 'rb') as f:
            sample_names, labels = pickle.load(f)

        # Apply max_samples limit
        num_samples = len(labels)
        if max_samples and max_samples < num_samples:
            print(f"Limiting to {max_samples} samples (total available: {num_samples})")
            data = data[:max_samples]
            labels = labels[:max_samples]
            num_samples = max_samples

        print(f"Processing {num_samples} samples with batch_size={batch_size}")

        # Run inference
        predictions = []
        inference_times = []

        num_batches = (len(data) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(data))

            batch_data = data[start_idx:end_idx]
            batch_data = self.preprocess(batch_data)

            # Inference
            start_time = time.perf_counter()
            output = self.engine.infer(batch_data)
            inference_time = (time.perf_counter() - start_time) * 1000

            # Get predictions
            batch_predictions = np.argmax(output, axis=1)
            predictions.extend(batch_predictions.tolist())
            inference_times.append(inference_time)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Progress: {batch_idx + 1}/{num_batches} batches")

        predictions = np.array(predictions)
        labels = np.array(labels)

        # Calculate accuracy
        top1_acc = np.mean(predictions == labels)

        # Calculate top-5 accuracy (use already loaded data with max_samples applied)
        print("\nCalculating top-5 accuracy...")
        top5_correct = 0
        num_samples = len(labels)

        for i in range(num_samples):
            sample_data = self.preprocess(data[i])
            output = self.engine.infer(sample_data)
            top5_pred = np.argsort(output[0])[-5:]
            if labels[i] in top5_pred:
                top5_correct += 1

            if (i + 1) % 100 == 0:
                print(f"  Top-5 progress: {i + 1}/{num_samples}")

        top5_acc = top5_correct / num_samples

        # Calculate timing statistics
        total_time = sum(inference_times)
        avg_batch_time = np.mean(inference_times)
        avg_sample_latency = total_time / num_samples  # Average latency per sample
        throughput = num_samples / (total_time / 1000)  # samples per second

        results = {
            'num_samples': num_samples,
            'batch_size': batch_size,
            'top1_accuracy': float(top1_acc),
            'top5_accuracy': float(top5_acc),
            'total_time_ms': float(total_time),
            'avg_batch_time_ms': float(avg_batch_time),
            'avg_sample_latency_ms': float(avg_sample_latency),
            'throughput_samples_per_sec': float(throughput)
        }

        print("\n=== Evaluation Results ===")
        print(f"Samples: {results['num_samples']} (batch_size={results['batch_size']})")
        print(f"Top-1 Accuracy: {results['top1_accuracy']:.2%}")
        print(f"Top-5 Accuracy: {results['top5_accuracy']:.2%}")
        print(f"\n=== Performance ===")
        print(f"Total Time: {results['total_time_ms']:.2f} ms")
        print(f"Avg Batch Time: {results['avg_batch_time_ms']:.2f} ms")
        print(f"Avg Sample Latency: {results['avg_sample_latency_ms']:.2f} ms")
        print(f"Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")

        return results


def load_label_map(dataset='ntu-xsub'):
    """
    Load label map for the dataset

    Args:
        dataset: Dataset name

    Returns:
        dict: Label map
    """
    # NTU RGB+D 60 action classes
    ntu_60_labels = {
        0: 'drink water', 1: 'eat meal/snack', 2: 'brushing teeth',
        3: 'brushing hair', 4: 'drop', 5: 'pickup',
        6: 'throw', 7: 'sitting down', 8: 'standing up (from sitting position)',
        9: 'clapping', 10: 'reading', 11: 'writing',
        12: 'tear up paper', 13: 'wear jacket', 14: 'take off jacket',
        15: 'wear a shoe', 16: 'take off a shoe', 17: 'wear on glasses',
        18: 'take off glasses', 19: 'put on a hat/cap', 20: 'take off a hat/cap',
        21: 'cheer up', 22: 'hand waving', 23: 'kicking something',
        24: 'reach into pocket', 25: 'hopping (one foot jumping)', 26: 'jump up',
        27: 'make a phone call/answer phone', 28: 'playing with phone/tablet',
        29: 'typing on a keyboard', 30: 'pointing to something with finger',
        31: 'taking a selfie', 32: 'check time (from watch)', 33: 'rub two hands together',
        34: 'nod head/bow', 35: 'shake head', 36: 'wipe face',
        37: 'salute', 38: 'put the palms together', 39: 'cross hands in front (say stop)',
        40: 'sneeze/cough', 41: 'staggering', 42: 'falling',
        43: 'touch head (headache)', 44: 'touch chest (stomachache/heart pain)',
        45: 'touch back (backache)', 46: 'touch neck (neckache)', 47: 'nausea or vomiting condition',
        48: 'use a fan (with hand or paper)/feeling warm', 49: 'punching/slapping other person',
        50: 'kicking other person', 51: 'pushing other person', 52: 'pat on back of other person',
        53: 'point finger at the other person', 54: 'hugging other person', 55: 'giving something to other person',
        56: 'touch other person\'s pocket', 57: 'handshaking', 58: 'walking towards each other',
        59: 'walking apart from each other'
    }

    if 'ntu' in dataset.lower():
        return ntu_60_labels
    else:
        return {}


def main():
    parser = argparse.ArgumentParser(
        description='OpenVINO End-to-End Inference Pipeline'
    )
    parser.add_argument('--model-xml', type=str, required=True,
                        help='Path to OpenVINO IR XML file')
    parser.add_argument('--mode', type=str, default='predict',
                        choices=['predict', 'evaluate', 'benchmark'],
                        help='Pipeline mode: predict, evaluate, or benchmark')
    parser.add_argument('--data-path', type=str, default='',
                        help='Path to test data (.npy file)')
    parser.add_argument('--label-path', type=str, default='',
                        help='Path to test labels (.pkl file)')
    parser.add_argument('--dataset', type=str, default='ntu-xsub',
                        help='Dataset name for label mapping')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='Number of iterations for benchmark mode')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations for benchmark mode')
    parser.add_argument('--device', type=str, default='GPU',
                        help='OpenVINO device (CPU, GPU, AUTO)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                        help='Output directory')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load label map
    label_map = load_label_map(args.dataset)

    # Initialize pipeline
    print("=== Initializing OpenVINO Pipeline ===")
    pipeline = OpenVINOActionRecognitionPipeline(
        args.model_xml,
        label_map=label_map,
        device=args.device
    )

    if args.mode == 'predict':
        # Prediction mode
        print("\n=== Running Prediction ===")

        # Load or generate sample data
        if args.data_path and os.path.exists(args.data_path):
            print(f"Loading data from: {args.data_path}")
            data = np.load(args.data_path, mmap_mode='r')
            sample = data[0]
        else:
            print("Generating random sample data...")
            sample = np.random.randn(3, 300, 25, 2).astype(np.float32)

        # Run prediction
        result = pipeline.predict(sample, top_k=5)

        print("\n=== Prediction Results ===")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        print(f"\nTop prediction:")
        print(f"  Action: {result['top_prediction']['action']}")
        print(f"  Probability: {result['top_prediction']['probability']:.4f}")
        print(f"\nTop-5 predictions:")
        for i, pred in enumerate(result['top_k_predictions'], 1):
            print(f"  {i}. {pred['action']}: {pred['probability']:.4f}")

        # Save results
        output_file = os.path.join(args.output_dir, 'prediction_result.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    elif args.mode == 'evaluate':
        # Evaluation mode
        if not args.data_path or not args.label_path:
            print("ERROR: --data-path and --label-path are required for evaluation mode")
            sys.exit(1)

        # Run evaluation
        results = pipeline.evaluate(
            args.data_path,
            args.label_path,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )

        # Save results
        output_file = os.path.join(args.output_dir, 'evaluation_result.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    elif args.mode == 'benchmark':
        # Benchmark mode - test multi-round inference latency
        print("\n=== Running Benchmark Mode ===")

        # Run benchmark
        results = pipeline.benchmark(
            data_path=args.data_path if args.data_path else None,
            num_iterations=args.num_iterations,
            warmup=args.warmup,
            max_samples=args.max_samples
        )

        # Save results
        output_file = os.path.join(args.output_dir, 'benchmark_result.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    print("\n=== Pipeline Complete! ===")


if __name__ == '__main__':
    main()
