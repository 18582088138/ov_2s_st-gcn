#!/usr/bin/env python
"""
Export 2S-STGCN PyTorch model to OpenVINO IR format
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.st_gcn_twostream import Model as TwoStreamModel
from net.utils.graph import Graph


def export_to_onnx(model, onnx_path, input_shape=(1, 3, 300, 25, 2), opset_version=None):
    """
    Export PyTorch model to ONNX format with smart opset version selection

    Args:
        model: PyTorch model
        onnx_path: Path to save ONNX model
        input_shape: Input tensor shape (N, C, T, V, M)
        opset_version: ONNX opset version (None for auto-detect)
    """
    model.eval()

    # Auto-detect opset version if not specified
    if opset_version is None:
        pytorch_version = torch.__version__
        major = int(pytorch_version.split('.')[0])
        minor = int(pytorch_version.split('.')[1].split('+')[0] if '+' in pytorch_version.split('.')[1] else pytorch_version.split('.')[1])

        # PyTorch 2.x → opset 17-18
        # PyTorch 1.13+ → opset 16
        # PyTorch 1.8-1.12 → opset 13
        if major >= 2:
            opset_version = 17  # Recommended for PyTorch 2.x
        elif minor >= 13:
            opset_version = 16
        else:
            opset_version = 13

        print(f"Auto-detected ONNX opset version: {opset_version} (PyTorch {pytorch_version})")
    else:
        print(f"Using specified ONNX opset version: {opset_version}")

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export to ONNX
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"✓ ONNX model saved to: {onnx_path}")

        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            actual_opset = onnx_model.opset_import[0].version
            print(f"✓ ONNX model verification passed (opset: {actual_opset})")
        except ImportError:
            print("  (onnx package not available for verification)")
        except Exception as e:
            print(f"  Warning: ONNX verification failed: {e}")

    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        raise


def convert_onnx_to_openvino(onnx_path, output_dir):
    """
    Convert ONNX model to OpenVINO IR format using OpenVINO 2.0 API

    Args:
        onnx_path: Path to ONNX model
        output_dir: Directory to save OpenVINO IR files

    Returns:
        tuple: (xml_path, bin_path)
    """
    try:
        import openvino as ov

        print(f"Converting ONNX to OpenVINO IR...")
        print(f"OpenVINO version: {ov.__version__}")

        # Convert ONNX to OpenVINO IR using OpenVINO 2.0 API
        ov_model = ov.convert_model(onnx_path)

        # Generate output paths
        model_name = os.path.splitext(os.path.basename(onnx_path))[0]
        xml_path = os.path.join(output_dir, f"{model_name}.xml")
        bin_path = os.path.join(output_dir, f"{model_name}.bin")

        # Save model (OpenVINO 2.0 API)
        ov.save_model(ov_model, xml_path)

        print(f"OpenVINO IR saved to: {xml_path}")
        print(f"OpenVINO weights saved to: {bin_path}")

        return xml_path, bin_path

    except ImportError as e:
        print(f"ERROR: OpenVINO not installed: {e}")
        print("Please install: pip install openvino==2026.2.0")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        raise


def load_pretrained_model(weights_path, num_class=60, layout='ntu-rgb+d'):
    """
    Load pretrained 2S-STGCN model

    Args:
        weights_path: Path to .pt weights file
        num_class: Number of action classes
        layout: Skeleton layout type

    Returns:
        Loaded model
    """
    # Model configuration
    graph_args = {
        'layout': layout,
        'strategy': 'spatial'
    }

    # Create model
    model = TwoStreamModel(
        in_channels=3,
        num_class=num_class,
        graph_args=graph_args,
        edge_importance_weighting=True
    )

    # Load weights
    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        print("Weights loaded successfully!")
    else:
        print(f"WARNING: Weights file not found: {weights_path}")
        print("Using randomly initialized model for export demo")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Export 2S-STGCN to OpenVINO')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to pretrained .pt weights file')
    parser.add_argument('--output-dir', type=str, default='./export_models',
                        help='Output directory for exported models')
    parser.add_argument('--num-class', type=int, default=60,
                        help='Number of action classes')
    parser.add_argument('--layout', type=str, default='ntu-rgb+d',
                        choices=['ntu-rgb+d', 'openpose', 'ntu_edge'],
                        help='Skeleton layout')
    parser.add_argument('--input-shape', type=str, default='1,3,300,25,2',
                        help='Input shape as comma-separated values (N,C,T,V,M)')
    parser.add_argument('--opset-version', type=int, default=None,
                        help='ONNX opset version (default: auto-detect based on PyTorch version)')

    args = parser.parse_args()

    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    print(f"Input shape: {input_shape}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained model
    print("\n=== Step 1: Loading PyTorch Model ===")
    model = load_pretrained_model(args.weights, args.num_class, args.layout)

    # Export to ONNX
    print("\n=== Step 2: Exporting to ONNX ===")
    model_name = os.path.splitext(os.path.basename(args.weights))[0]
    onnx_path = os.path.join(args.output_dir, f"{model_name}.onnx")
    export_to_onnx(model, onnx_path, input_shape, opset_version=args.opset_version)

    # Convert to OpenVINO
    print("\n=== Step 3: Converting to OpenVINO IR ===")
    xml_path, bin_path = convert_onnx_to_openvino(onnx_path, args.output_dir)

    print("\n=== Export Complete! ===")
    print(f"ONNX model: {onnx_path}")
    print(f"OpenVINO IR: {xml_path}")
    print(f"OpenVINO weights: {bin_path}")


if __name__ == '__main__':
    main()
