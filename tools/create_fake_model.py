#!/usr/bin/env python
"""
Create a fake pretrained model for testing OpenVINO pipeline

This script creates a 2S-STGCN model with random weights and saves it
in the same format as real pretrained models. Useful for:
- Testing the OpenVINO conversion pipeline
- Demonstrating the workflow without training
- Quick validation of export and inference tools

Usage:
    python tools/create_fake_model.py --output fake_model.pt --num-class 60
"""

import os
import sys
import argparse
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from net.st_gcn_twostream import Model as TwoStreamModel


def create_fake_model(num_class=60, layout='ntu-rgb+d'):
    """
    Create a 2S-STGCN model with random initialization

    Args:
        num_class: Number of action classes
        layout: Skeleton layout type

    Returns:
        model: Initialized model
    """
    print(f"Creating 2S-STGCN model...")
    print(f"  Num classes: {num_class}")
    print(f"  Layout: {layout}")

    model = TwoStreamModel(
        in_channels=3,
        num_class=num_class,
        graph_args={'layout': layout, 'strategy': 'spatial'},
        edge_importance_weighting=True
    )

    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


def save_fake_checkpoint(model, output_path, format_type='model_state_dict'):
    """
    Save model in checkpoint format

    Args:
        model: PyTorch model
        output_path: Path to save checkpoint
        format_type: One of 'model_state_dict', 'state_dict', 'direct'
    """
    print(f"\nSaving checkpoint...")
    print(f"  Format: {format_type}")
    print(f"  Output: {output_path}")

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    if format_type == 'model_state_dict':
        # Most common format: {'model_state_dict': state_dict, ...}
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 80,
            'optimizer_state_dict': {},  # Empty for fake model
            'best_acc': 0.89,  # Fake accuracy
        }
    elif format_type == 'state_dict':
        # Alternative format: {'state_dict': state_dict, ...}
        checkpoint = {
            'state_dict': model.state_dict(),
            'epoch': 80,
            'best_acc': 0.89,
        }
    elif format_type == 'direct':
        # Direct state dict
        checkpoint = model.state_dict()
    else:
        raise ValueError(f"Unknown format type: {format_type}")

    torch.save(checkpoint, output_path)

    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"  File size: {file_size:.2f} MB")
    print(f"✓ Checkpoint saved successfully!")


def verify_checkpoint(checkpoint_path, num_class=60, layout='ntu-rgb+d'):
    """
    Verify the saved checkpoint can be loaded

    Args:
        checkpoint_path: Path to checkpoint file
        num_class: Number of classes
        layout: Skeleton layout
    """
    print(f"\nVerifying checkpoint...")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Detect format
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                format_type = 'model_state_dict'
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                format_type = 'state_dict'
            else:
                state_dict = checkpoint
                format_type = 'direct'
        else:
            state_dict = checkpoint
            format_type = 'direct'

        print(f"  Format detected: {format_type}")
        print(f"  Number of state dict keys: {len(state_dict)}")

        # Create model and load weights
        model = TwoStreamModel(
            in_channels=3,
            num_class=num_class,
            graph_args={'layout': layout, 'strategy': 'spatial'},
            edge_importance_weighting=True
        )

        model.load_state_dict(state_dict)
        model.eval()

        # Test inference
        test_input = torch.randn(1, 3, 300, 25, 2)
        with torch.no_grad():
            output = model(test_input)

        print(f"  Model loaded successfully!")
        print(f"  Test output shape: {output.shape}")
        print(f"✓ Verification passed!")

        return True

    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Create fake pretrained model for testing')

    parser.add_argument('--output', type=str, default='./fake_models/fake_model.pt',
                        help='Output path for the fake model')
    parser.add_argument('--num-class', type=int, default=60,
                        help='Number of action classes (default: 60 for NTU RGB+D)')
    parser.add_argument('--layout', type=str, default='ntu-rgb+d',
                        choices=['ntu-rgb+d', 'openpose', 'ntu-edge'],
                        help='Skeleton layout type')
    parser.add_argument('--format', type=str, default='model_state_dict',
                        choices=['model_state_dict', 'state_dict', 'direct'],
                        help='Checkpoint format type')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification after saving')

    args = parser.parse_args()

    print("="*70)
    print("Create Fake Pretrained Model")
    print("="*70)

    # Create model
    model = create_fake_model(num_class=args.num_class, layout=args.layout)

    # Save checkpoint
    save_fake_checkpoint(model, args.output, format_type=args.format)

    # Verify
    if not args.no_verify:
        verify_checkpoint(args.output, num_class=args.num_class, layout=args.layout)

    # Usage instructions
    print("\n" + "="*70)
    print("Next Steps")
    print("="*70)
    print("\n1. Export to OpenVINO:")
    print(f"   python tools/export_to_openvino.py \\")
    print(f"       --weights {args.output} \\")
    print(f"       --output-dir ./export_models \\")
    print(f"       --num-class {args.num_class}")

    print("\n2. Compare inference:")
    print(f"   python tools/compare_inference.py \\")
    print(f"       --weights {args.output} \\")
    print(f"       --openvino-xml ./export_models/fake_model.xml")

    print("\n3. Run E2E pipeline:")
    print(f"   python tools/openvino_e2e_pipeline.py \\")
    print(f"       --model-xml ./export_models/fake_model.xml \\")
    print(f"       --mode predict")

    print("\n4. Run complete test:")
    print(f"   python tests/run_full_pipeline.py \\")
    print(f"       --weights {args.output}")

    print("\n" + "="*70)
    print("✓ Fake model created successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
