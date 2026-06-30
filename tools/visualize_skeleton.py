#!/usr/bin/env python
"""
Skeleton visualization tool for 2S-STGCN
Visualize skeleton sequences and save as images/videos
"""
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# NTU RGB+D skeleton connections (25 joints)
NTU_SKELETON_BONE = [
    (0, 1), (1, 20), (20, 2), (2, 3),  # Spine
    (20, 4), (4, 5), (5, 6), (6, 7), (7, 21), (7, 22),  # Left arm
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 23), (11, 24),  # Right arm
    (0, 12), (12, 13), (13, 14), (14, 15),  # Left leg
    (0, 16), (16, 17), (17, 18), (18, 19),  # Right leg
]

# Kinetics skeleton connections (18 joints)
KINETICS_SKELETON_BONE = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Head to shoulder
    (1, 5), (5, 6), (6, 7),  # Left arm
    (1, 8), (8, 9), (9, 10),  # Right arm
    (1, 11), (11, 12), (12, 13),  # Left leg
    (1, 14), (14, 15), (15, 16),  # Right leg
]

# Action labels for NTU RGB+D 60 classes
NTU_ACTION_LABELS = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop",
    "pickup", "throw", "sitting down", "standing up (from sitting position)", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket",
    "hopping (one foot jumping)", "jump up", "make a phone call/answer phone", "playing with phone/tablet", "typing on a keyboard",
    "pointing to something with finger", "taking a selfie", "check time (from watch)", "rub two hands together", "nod head/bow",
    "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (say stop)",
    "sneeze/cough", "staggering", "falling", "touch head (headache)", "touch chest (stomachache/heart pain)",
    "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person", "point finger at the other person", "hugging other person",
    "giving something to other person", "touch other person's pocket", "handshaking", "walking towards each other", "walking apart from each other"
]


def visualize_skeleton_frame(skeleton_data, frame_idx, skeleton_type='ntu-rgb+d',
                            action_label=None, save_path=None):
    """
    Visualize a single frame of skeleton data

    Args:
        skeleton_data: Skeleton data (C, T, V, M) or (T, V, M, C)
        frame_idx: Frame index to visualize
        skeleton_type: 'ntu-rgb+d' or 'kinetics'
        action_label: Action label text
        save_path: Path to save the figure
    """
    # Parse skeleton data
    if skeleton_data.shape[0] == 3:  # (C, T, V, M)
        skeleton_data = skeleton_data.transpose(1, 2, 3, 0)  # (T, V, M, C)

    T, V, M, C = skeleton_data.shape

    if frame_idx >= T:
        frame_idx = T - 1

    # Get frame data
    frame_data = skeleton_data[frame_idx]  # (V, M, C)

    # Select skeleton connections
    if skeleton_type == 'ntu-rgb+d':
        bones = NTU_SKELETON_BONE
    else:
        bones = KINETICS_SKELETON_BONE

    # Create figure
    fig = plt.figure(figsize=(12, 6))

    # Plot each person
    colors = ['red', 'blue', 'green']
    for person_idx in range(M):
        ax = fig.add_subplot(1, M, person_idx + 1, projection='3d')

        joints = frame_data[:, person_idx, :]  # (V, C)

        # Check if this person exists (non-zero joints)
        if np.sum(np.abs(joints)) < 1e-5:
            ax.set_title(f'Person {person_idx + 1}: Not present')
            continue

        # Plot joints
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                  c=colors[person_idx % len(colors)], marker='o', s=50)

        # Plot bones
        for bone in bones:
            if bone[0] < V and bone[1] < V:
                start_joint = joints[bone[0]]
                end_joint = joints[bone[1]]

                # Only draw if both joints exist
                if np.sum(np.abs(start_joint)) > 1e-5 and np.sum(np.abs(end_joint)) > 1e-5:
                    ax.plot([start_joint[0], end_joint[0]],
                           [start_joint[1], end_joint[1]],
                           [start_joint[2], end_joint[2]],
                           c=colors[person_idx % len(colors)], linewidth=2)

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Person {person_idx + 1}')

        # Set axis limits
        max_range = np.max(np.abs(joints)) * 1.2
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

    # Add title with action label
    if action_label is not None:
        fig.suptitle(f'Frame {frame_idx} - Action: {action_label}', fontsize=14)
    else:
        fig.suptitle(f'Frame {frame_idx}', fontsize=14)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved frame to: {save_path}")
        plt.close()
    else:
        plt.show()


def visualize_skeleton_sequence(skeleton_data, skeleton_type='ntu-rgb+d',
                               action_label=None, save_path=None,
                               frame_interval=10, max_frames=30):
    """
    Visualize a sequence of skeleton frames

    Args:
        skeleton_data: Skeleton data (C, T, V, M) or (T, V, M, C)
        skeleton_type: 'ntu-rgb+d' or 'kinetics'
        action_label: Action label text
        save_path: Directory to save frame images
        frame_interval: Interval between saved frames
        max_frames: Maximum number of frames to save
    """
    # Parse skeleton data
    if skeleton_data.shape[0] == 3:  # (C, T, V, M)
        skeleton_data = skeleton_data.transpose(1, 2, 3, 0)  # (T, V, M, C)

    T, V, M, C = skeleton_data.shape

    if save_path:
        os.makedirs(save_path, exist_ok=True)

    # Select frames to visualize
    frame_indices = list(range(0, T, frame_interval))[:max_frames]

    print(f"Visualizing {len(frame_indices)} frames (interval={frame_interval})...")

    for i, frame_idx in enumerate(frame_indices):
        if save_path:
            frame_save_path = os.path.join(save_path, f'frame_{frame_idx:04d}.png')
        else:
            frame_save_path = None

        visualize_skeleton_frame(
            skeleton_data, frame_idx,
            skeleton_type=skeleton_type,
            action_label=action_label,
            save_path=frame_save_path
        )

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(frame_indices)}")

    print(f"✓ Saved {len(frame_indices)} frames to: {save_path}")


def create_skeleton_video(skeleton_data, skeleton_type='ntu-rgb+d',
                         action_label=None, save_path='skeleton.mp4', fps=30):
    """
    Create a video animation of skeleton sequence

    Args:
        skeleton_data: Skeleton data (C, T, V, M) or (T, V, M, C)
        skeleton_type: 'ntu-rgb+d' or 'kinetics'
        action_label: Action label text
        save_path: Path to save video
        fps: Frames per second
    """
    print(f"Creating video: {save_path}")

    # Parse skeleton data
    if skeleton_data.shape[0] == 3:  # (C, T, V, M)
        skeleton_data = skeleton_data.transpose(1, 2, 3, 0)  # (T, V, M, C)

    T, V, M, C = skeleton_data.shape

    # Select skeleton connections
    if skeleton_type == 'ntu-rgb+d':
        bones = NTU_SKELETON_BONE
    else:
        bones = KINETICS_SKELETON_BONE

    # Create figure
    fig = plt.figure(figsize=(12, 6))
    axes = []

    colors = ['red', 'blue', 'green']

    # Initialize subplots for each person
    for person_idx in range(M):
        ax = fig.add_subplot(1, M, person_idx + 1, projection='3d')
        axes.append(ax)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Person {person_idx + 1}')

    # Set title
    if action_label is not None:
        fig.suptitle(f'Action: {action_label}', fontsize=14)

    def update_frame(frame_idx):
        """Update function for animation"""
        frame_data = skeleton_data[frame_idx]  # (V, M, C)

        for person_idx, ax in enumerate(axes):
            ax.cla()
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Person {person_idx + 1} - Frame {frame_idx}')

            joints = frame_data[:, person_idx, :]  # (V, C)

            # Check if this person exists
            if np.sum(np.abs(joints)) < 1e-5:
                continue

            # Plot joints
            ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2],
                      c=colors[person_idx % len(colors)], marker='o', s=50)

            # Plot bones
            for bone in bones:
                if bone[0] < V and bone[1] < V:
                    start_joint = joints[bone[0]]
                    end_joint = joints[bone[1]]

                    if np.sum(np.abs(start_joint)) > 1e-5 and np.sum(np.abs(end_joint)) > 1e-5:
                        ax.plot([start_joint[0], end_joint[0]],
                               [start_joint[1], end_joint[1]],
                               [start_joint[2], end_joint[2]],
                               c=colors[person_idx % len(colors)], linewidth=2)

            # Set axis limits
            max_range = np.max(np.abs(skeleton_data)) * 1.2
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

    # Create animation
    ani = FuncAnimation(fig, update_frame, frames=T, interval=1000/fps, repeat=True)

    # Save video
    try:
        writer = FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(save_path, writer=writer)
        print(f"✓ Video saved to: {save_path}")
    except Exception as e:
        print(f"✗ Failed to save video: {e}")
        print("  Note: ffmpeg is required for video export")
        print("  Install: conda install ffmpeg or apt-get install ffmpeg")
    finally:
        plt.close()


def visualize_predictions(skeleton_data, predictions, skeleton_type='ntu-rgb+d',
                         save_dir='./outputs/visualizations', top_k=5):
    """
    Visualize skeleton with predictions

    Args:
        skeleton_data: Skeleton data (C, T, V, M)
        predictions: Prediction scores (num_classes,)
        skeleton_type: Skeleton type
        save_dir: Output directory
        top_k: Number of top predictions to show
    """
    os.makedirs(save_dir, exist_ok=True)

    # Get top-k predictions
    top_k_indices = np.argsort(predictions)[-top_k:][::-1]
    top_k_scores = predictions[top_k_indices]

    # Get action labels
    if len(predictions) == 60:
        action_labels = NTU_ACTION_LABELS
    else:
        action_labels = [f"Class {i}" for i in range(len(predictions))]

    # Create prediction text
    pred_text = "Top-{} Predictions:\n".format(top_k)
    for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
        pred_text += f"  {i+1}. {action_labels[idx]}: {score:.3f}\n"

    print(pred_text)

    # Visualize skeleton with top prediction
    top_action = action_labels[top_k_indices[0]]

    # Save frames
    frame_dir = os.path.join(save_dir, 'frames')
    visualize_skeleton_sequence(
        skeleton_data,
        skeleton_type=skeleton_type,
        action_label=f"{top_action} ({top_k_scores[0]:.3f})",
        save_path=frame_dir,
        frame_interval=15,
        max_frames=20
    )

    # Save prediction text
    pred_file = os.path.join(save_dir, 'predictions.txt')
    with open(pred_file, 'w') as f:
        f.write(pred_text)
    print(f"✓ Predictions saved to: {pred_file}")


def main():
    parser = argparse.ArgumentParser(description='Visualize skeleton data')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to skeleton data (.npy file with shape C,T,V,M or T,V,M,C)')
    parser.add_argument('--skeleton-type', type=str, default='ntu-rgb+d',
                        choices=['ntu-rgb+d', 'kinetics'],
                        help='Skeleton type')
    parser.add_argument('--output-dir', type=str, default='./outputs/visualizations',
                        help='Output directory')
    parser.add_argument('--frame-interval', type=int, default=10,
                        help='Interval between saved frames')
    parser.add_argument('--max-frames', type=int, default=30,
                        help='Maximum number of frames to save')
    parser.add_argument('--action-label', type=str, default=None,
                        help='Action label text')
    parser.add_argument('--create-video', action='store_true',
                        help='Create video animation (requires ffmpeg)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video frame rate')

    args = parser.parse_args()

    # Load skeleton data
    print(f"Loading skeleton data from: {args.data}")
    skeleton_data = np.load(args.data)
    print(f"Data shape: {skeleton_data.shape}")

    # Handle different input shapes
    if len(skeleton_data.shape) == 5:
        # (N, C, T, V, M) - take first sample
        skeleton_data = skeleton_data[0]
        print(f"Using first sample, shape: {skeleton_data.shape}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Visualize frames
    print("\n=== Visualizing Frames ===")
    frame_dir = os.path.join(args.output_dir, 'frames')
    visualize_skeleton_sequence(
        skeleton_data,
        skeleton_type=args.skeleton_type,
        action_label=args.action_label,
        save_path=frame_dir,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames
    )

    # Create video if requested
    if args.create_video:
        print("\n=== Creating Video ===")
        video_path = os.path.join(args.output_dir, 'skeleton_animation.mp4')
        create_skeleton_video(
            skeleton_data,
            skeleton_type=args.skeleton_type,
            action_label=args.action_label,
            save_path=video_path,
            fps=args.fps
        )

    print("\n=== Visualization Complete! ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
