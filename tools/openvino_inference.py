#!/usr/bin/env python
"""
OpenVINO Inference Engine for 2S-STGCN
"""
import os
import time
import numpy as np


class OpenVINOInferenceEngine:
    """
    OpenVINO inference engine for skeleton-based action recognition
    """

    def __init__(self, model_xml, device='GPU'):
        """
        Initialize OpenVINO inference engine using OpenVINO 2.0 API

        Args:
            model_xml: Path to OpenVINO IR XML file
            device: Target device (CPU, GPU, AUTO, etc.)
        """
        try:
            import openvino as ov
        except ImportError:
            raise ImportError(
                "OpenVINO not installed. Install: pip install openvino==2026.2.0"
            )

        self.device = device
        self.model_xml = model_xml

        # Initialize OpenVINO Core (OpenVINO 2.0 API)
        print(f"Initializing OpenVINO {ov.__version__} on device: {device}")
        self.core = ov.Core()

        # Check available devices
        self._check_device_availability()

        # Read model
        print(f"Loading model from: {model_xml}")
        self.model = self.core.read_model(model=model_xml)

        # Compile model with performance hints
        print(f"Compiling model for {device}...")
        config = {}
        if 'GPU' in device.upper():
            # GPU-specific optimizations
            config['PERFORMANCE_HINT'] = 'THROUGHPUT'
            print("  GPU mode: Optimizing for throughput")
        self.compiled_model = self.core.compile_model(self.model, device, config)

        # Create inference request (CRITICAL for GPU sync!)
        self.infer_request = self.compiled_model.create_infer_request()

        # Get input/output info
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        print("✓ OpenVINO engine initialized successfully!")
        # print(f"  Input shape: {self.input_layer.shape}")
        # print(f"  Output shape: {self.output_layer.shape}")

    def _check_device_availability(self):
        """Check available devices and warn if GPU not found"""
        try:
            available_devices = self.core.available_devices
            print(f"\nAvailable OpenVINO devices: {available_devices}")

            for device in available_devices:
                try:
                    device_name = self.core.get_property(device, "FULL_DEVICE_NAME")
                    print(f"  - {device}: {device_name}")
                except:
                    print(f"  - {device}")

            # Check if requested device is available
            if 'GPU' in self.device.upper():
                has_gpu = any('GPU' in d for d in available_devices)
                if not has_gpu:
                    print("\n⚠️  WARNING: GPU device not found!")
                    print("OpenVINO GPU requires:")
                    print("  - Intel GPU: Intel Graphics Driver + OpenCL Runtime")
                    print("  - NVIDIA GPU: CUDA + OpenCL")
                    print("\nContinuing anyway, but inference may fall back to CPU.")
        except Exception as e:
            print(f"Warning: Could not check device availability: {e}")

    def infer(self, input_data):
        """
        Run synchronous inference on input data

        IMPORTANT: Uses infer_request.infer() to ensure GPU computation completes
        before returning. This is critical for accurate performance measurement.

        Args:
            input_data: Input numpy array with shape (N, C, T, V, M)

        Returns:
            Output predictions with shape (N, num_classes)
        """
        # Ensure input is numpy array
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        # Run synchronous inference (GPU/CPU computation completes before return)
        self.infer_request.infer({self.input_layer: input_data})

        # Get output tensor (computation is guaranteed complete here)
        result = self.infer_request.get_output_tensor(0).data

        return result.copy()  # Copy to avoid memory issues

    def infer_with_timing(self, input_data):
        """
        Run inference and measure execution time

        Args:
            input_data: Input numpy array

        Returns:
            tuple: (predictions, inference_time_ms)
        """
        start_time = time.perf_counter()
        result = self.infer(input_data)
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

        return result, inference_time

    def benchmark(self, input_data, num_iterations=100, warmup=10):
        """
        Benchmark inference performance with proper GPU warmup

        Args:
            input_data: Input numpy array
            num_iterations: Number of inference iterations
            warmup: Number of warmup iterations

        Returns:
            dict: Performance statistics
        """
        batch_size = input_data.shape[0]
        print(f"\nBenchmarking OpenVINO inference on {self.device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Iterations: {num_iterations}")
        print(f"  Warmup: {warmup}")

        # Initial warmup
        print(f"Warmup phase 1: {warmup} iterations...")
        for _ in range(warmup):
            self.infer(input_data)

        # Extended GPU warmup (critical for GPU performance!)
        if 'GPU' in self.device.upper():
            gpu_warmup = 20
            print(f"Warmup phase 2 (GPU): {gpu_warmup} iterations...")
            for _ in range(gpu_warmup):
                self.infer(input_data)

        # Benchmark
        print(f"Running benchmark...")
        times = []
        for i in range(num_iterations):
            _, inference_time = self.infer_with_timing(input_data)
            times.append(inference_time)

            if (i + 1) % 20 == 0:
                avg_so_far = np.mean(times)
                print(f"  Progress: {i + 1}/{num_iterations} - Avg: {avg_so_far:.2f}ms")

        times = np.array(times)

        # Calculate throughput (samples per second)
        throughput = (batch_size * 1000.0) / np.mean(times)

        stats = {
            'device': self.device,
            'batch_size': batch_size,
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'fps': 1000.0 / np.mean(times),  # Batches per second
            'throughput': throughput  # Samples per second
        }

        print("\n=== Benchmark Results ===")
        print(f"Device: {stats['device']}")
        print(f"Batch size: {stats['batch_size']}")
        print(f"Mean: {stats['mean']:.2f} ms")
        print(f"Std:  {stats['std']:.2f} ms")
        print(f"Min:  {stats['min']:.2f} ms")
        print(f"Max:  {stats['max']:.2f} ms")
        print(f"Median: {stats['median']:.2f} ms")
        print(f"P95:  {stats['p95']:.2f} ms")
        print(f"P99:  {stats['p99']:.2f} ms")
        print(f"Throughput: {stats['throughput']:.2f} samples/sec")
        print(f"Latency per sample: {stats['mean']/batch_size:.2f} ms")

        return stats

    def get_model_info(self):
        """
        Get model information

        Returns:
            dict: Model information
        """
        return {
            'model_path': self.model_xml,
            'device': self.device,
            'input_shape': self.input_layer.shape,
            'input_dtype': self.input_layer.element_type,
            'output_shape': self.output_layer.shape,
            'output_dtype': self.output_layer.element_type
        }


class TorchInferenceEngine:
    """
    PyTorch inference engine for comparison
    """

    def __init__(self, model):
        """
        Initialize PyTorch inference engine

        Args:
            model: PyTorch model
        """
        import torch

        self.model = model
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        print(f"PyTorch model loaded on: {self.device}")

    def infer(self, input_data):
        """
        Run inference on input data

        Args:
            input_data: Input numpy array with shape (N, C, T, V, M)

        Returns:
            Output predictions with shape (N, num_classes)
        """
        import torch

        # Convert to torch tensor
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        else:
            input_tensor = input_data.float()

        input_tensor = input_tensor.to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)

        return output.cpu().numpy()

    def infer_with_timing(self, input_data):
        """
        Run inference and measure execution time

        Args:
            input_data: Input numpy array

        Returns:
            tuple: (predictions, inference_time_ms)
        """
        import torch

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        result = self.infer(input_data)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        inference_time = (end_time - start_time) * 1000

        return result, inference_time

    def benchmark(self, input_data, num_iterations=100, warmup=10):
        """
        Benchmark inference performance

        Args:
            input_data: Input numpy array
            num_iterations: Number of inference iterations
            warmup: Number of warmup iterations

        Returns:
            dict: Performance statistics
        """
        import torch

        print(f"\nBenchmarking PyTorch inference ({num_iterations} iterations)...")

        # Warmup
        print(f"Warmup: {warmup} iterations...")
        for _ in range(warmup):
            self.infer(input_data)

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Benchmark
        print(f"Running benchmark...")
        times = []
        for i in range(num_iterations):
            _, inference_time = self.infer_with_timing(input_data)
            times.append(inference_time)

            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{num_iterations}")

        times = np.array(times)

        # Calculate throughput (samples per second)
        batch_size = input_data.shape[0]
        throughput = (batch_size * 1000.0) / np.mean(times)

        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99),
            'fps': 1000.0 / np.mean(times),
            'throughput': throughput
        }

        print("\n=== Benchmark Results ===")
        print(f"Mean: {stats['mean']:.2f} ms")
        print(f"Std:  {stats['std']:.2f} ms")
        print(f"Min:  {stats['min']:.2f} ms")
        print(f"Max:  {stats['max']:.2f} ms")
        print(f"Median: {stats['median']:.2f} ms")
        print(f"P95:  {stats['p95']:.2f} ms")
        print(f"P99:  {stats['p99']:.2f} ms")
        print(f"FPS:  {stats['fps']:.2f}")
        print(f"Throughput: {stats['throughput']:.2f} samples/sec")
        print(f"Latency per sample: {stats['mean']/batch_size:.2f} ms")


        return stats
