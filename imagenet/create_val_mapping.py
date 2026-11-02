"""
اسکریپت ایجاد فایل mapping برای validation images
این اسکریپت فایل mapping را از فایل ground truth ImageNet ایجاد می‌کند
"""

import os
import sys
from pathlib import Path

def create_mapping_from_ground_truth(val_dir, ground_truth_file, output_file):
    """
    Create mapping file from ImageNet validation ground truth file
    
    Args:
        val_dir: Directory containing validation images
        ground_truth_file: Path to ILSVRC2012_validation_ground_truth.txt
        output_file: Output mapping file path
    """
    val_path = Path(val_dir)
    gt_path = Path(ground_truth_file)
    output_path = Path(output_file)
    
    # Get all validation images sorted by name
    val_images = sorted([f.name for f in val_path.glob('*.JPEG')] + [f.name for f in val_path.glob('*.jpg')])
    
    print(f"Found {len(val_images)} validation images")
    
    # Read ground truth file
    if not gt_path.exists():
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return False
    
    with open(gt_path, 'r') as f:
        gt_labels = [int(line.strip()) - 1 for line in f.readlines()]  # Convert to 0-based indexing
    
    print(f"Found {len(gt_labels)} ground truth labels")
    
    # Check if number of images matches number of labels
    if len(val_images) != len(gt_labels):
        print(f"Warning: Number of images ({len(val_images)}) does not match number of labels ({len(gt_labels)})")
        min_len = min(len(val_images), len(gt_labels))
        val_images = val_images[:min_len]
        gt_labels = gt_labels[:min_len]
        print(f"Using first {min_len} images and labels")
    
    # Create mapping file
    with open(output_path, 'w') as f:
        for img_name, label in zip(val_images, gt_labels):
            f.write(f"{img_name} {label}\n")
    
    print(f"Mapping file created: {output_file}")
    print(f"Total mappings: {len(val_images)}")
    
    return True

def create_mapping_from_synset_words(val_dir, synset_words_file, output_file):
    """
    Create mapping file using synset words file (alternative method)
    This requires additional processing and may not be as accurate
    """
    print("This method is not yet implemented. Please use create_mapping_from_ground_truth instead.")
    return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create validation mapping file')
    parser.add_argument('--val-dir', type=str, default='ILSVRC2012_img_val',
                        help='Directory containing validation images')
    parser.add_argument('--ground-truth', type=str, default='ILSVRC2012_validation_ground_truth.txt',
                        help='Path to ILSVRC2012_validation_ground_truth.txt file')
    parser.add_argument('--output', type=str, default='val_mapping.txt',
                        help='Output mapping file path')
    
    args = parser.parse_args()
    
    if create_mapping_from_ground_truth(args.val_dir, args.ground_truth, args.output):
        print("Success!")
        print(f"You can now use this mapping file with: --val-mapping {args.output}")
    else:
        print("Failed to create mapping file")
        sys.exit(1)



