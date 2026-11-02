"""
ImageNet Data Preparation Script
This script extracts tar files and organizes validation images to match the code structure.
"""

import os
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm

def extract_tar_files(train_dir, output_dir):
    """
    Extract all tar files from train_dir to output_dir
    Each tar file should contain images for one class
    """
    train_path = Path(train_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    tar_files = sorted(list(train_path.glob('*.tar')))
    print(f"Found {len(tar_files)} tar files")
    
    if len(tar_files) == 0:
        print(f"Error: No tar files found in {train_dir}")
        return False
    
    for tar_file in tqdm(tar_files, desc="Extracting tar files"):
        # Get class name from tar file name (remove .tar extension)
        class_name = tar_file.stem
        
        # Create class directory
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Extract tar file
        try:
            with tarfile.open(tar_file, 'r') as tar:
                # Extract to a temporary location first
                temp_dir = class_dir / '_temp'
                temp_dir.mkdir(exist_ok=True)
                tar.extractall(temp_dir)
                
                # Move files from subdirectories to class directory
                for item in temp_dir.rglob('*'):
                    if item.is_file():
                        # Move file to class directory
                        dest = class_dir / item.name
                        if not dest.exists():  # Avoid overwriting
                            shutil.move(str(item), str(dest))
                
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            print(f"Error extracting {tar_file}: {e}")
            continue
    
    print(f"Training data extracted to {output_dir}")
    return True

def organize_validation_with_ground_truth(val_dir, output_dir, ground_truth_file, class_names):
    """
    Organize validation images using ground truth file
    ground_truth_file: File with one label per line (1-indexed class IDs)
    class_names: List of class names in order (should match tar file names)
    """
    val_path = Path(val_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Read ground truth labels
    with open(ground_truth_file, 'r') as f:
        labels = [int(line.strip()) for line in f.readlines()]
    
    # Get sorted validation images
    val_images = sorted([f for f in val_path.glob('*.JPEG')] + [f for f in val_path.glob('*.jpg')])
    
    if len(val_images) != len(labels):
        print(f"Warning: Number of images ({len(val_images)}) != number of labels ({len(labels)})")
        min_len = min(len(val_images), len(labels))
        val_images = val_images[:min_len]
        labels = labels[:min_len]
    
    print(f"Organizing {len(val_images)} validation images into {len(class_names)} classes")
    
    # Create class directories
    for class_name in class_names:
        (output_path / class_name).mkdir(exist_ok=True)
    
    # Organize images
    for img_path, label in tqdm(zip(val_images, labels), desc="Organizing validation images", total=len(val_images)):
        # Convert 1-indexed to 0-indexed
        class_idx = label - 1
        if class_idx < 0 or class_idx >= len(class_names):
            print(f"Warning: Invalid label {label} for image {img_path.name}")
            continue
        
        class_name = class_names[class_idx]
        class_dir = output_path / class_name
        dest_path = class_dir / img_path.name
        
        # Copy image to class directory
        shutil.copy2(str(img_path), str(dest_path))
    
    print(f"Validation data organized to {output_dir}")
    return True

def organize_validation_with_mapping(val_dir, output_dir, mapping_file):
    """
    Organize validation images using a mapping file
    mapping_file: File with format: image_name class_name (one per line)
    """
    val_path = Path(val_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Read mapping file
    mappings = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_name = parts[0]
                class_name = parts[1]
                mappings[image_name] = class_name
    
    print(f"Found {len(mappings)} image mappings")
    
    # Create class directories and organize images
    val_images = list(val_path.glob('*.JPEG')) + list(val_path.glob('*.jpg'))
    organized = 0
    
    for img_path in tqdm(val_images, desc="Organizing validation images"):
        if img_path.name in mappings:
            class_name = mappings[img_path.name]
            class_dir = output_path / class_name
            class_dir.mkdir(exist_ok=True)
            dest_path = class_dir / img_path.name
            shutil.copy2(str(img_path), str(dest_path))
            organized += 1
        else:
            print(f"Warning: No mapping found for {img_path.name}")
    
    print(f"Organized {organized} validation images to {output_dir}")
    return organized > 0

def get_class_names_from_tar_dir(tar_dir):
    """Get sorted list of class names from tar files"""
    tar_path = Path(tar_dir)
    tar_files = sorted([f.stem for f in tar_path.glob('*.tar')])
    return tar_files

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare ImageNet data to match code structure')
    parser.add_argument('--train-tar-dir', type=str, default='ILSVRC2012_img_train',
                        help='Directory containing training tar files')
    parser.add_argument('--val-dir', type=str, default='ILSVRC2012_img_val',
                        help='Directory containing validation images')
    parser.add_argument('--output-train-dir', type=str, default='train',
                        help='Output directory for extracted training images')
    parser.add_argument('--output-val-dir', type=str, default='val',
                        help='Output directory for organized validation images')
    parser.add_argument('--ground-truth', type=str, default=None,
                        help='Path to ILSVRC2012_validation_ground_truth.txt file (1-indexed labels)')
    parser.add_argument('--val-mapping', type=str, default=None,
                        help='Path to validation mapping file (image_name class_name format)')
    parser.add_argument('--extract-train', action='store_true',
                        help='Extract training tar files')
    parser.add_argument('--organize-val', action='store_true',
                        help='Organize validation images')
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Base directory for output (train and val will be created here)')
    
    args = parser.parse_args()
    
    # Set output directories relative to data_dir
    train_output = Path(args.data_dir) / args.output_train_dir
    val_output = Path(args.data_dir) / args.output_val_dir
    
    print("=" * 60)
    print("ImageNet Data Preparation")
    print("=" * 60)
    
    if args.extract_train:
        print("\n[Step 1] Extracting training data...")
        extract_tar_files(args.train_tar_dir, train_output)
        print(f"[OK] Training data ready at: {train_output}")
    
    if args.organize_val:
        print("\n[Step 2] Organizing validation data...")
        
        if args.ground_truth and os.path.exists(args.ground_truth):
            # Use ground truth file
            class_names = get_class_names_from_tar_dir(args.train_tar_dir)
            if len(class_names) != 1000:
                print(f"Warning: Expected 1000 classes, found {len(class_names)}")
            organize_validation_with_ground_truth(args.val_dir, val_output, args.ground_truth, class_names)
            print(f"[OK] Validation data ready at: {val_output}")
        
        elif args.val_mapping and os.path.exists(args.val_mapping):
            # Use mapping file
            organize_validation_with_mapping(args.val_dir, val_output, args.val_mapping)
            print(f"[OK] Validation data ready at: {val_output}")
        
        else:
            print("\n[ERROR] Validation organization requires one of:")
            print("  1. --ground-truth: Path to ILSVRC2012_validation_ground_truth.txt")
            print("  2. --val-mapping: Path to validation mapping file")
            print("\nTo get the ground truth file:")
            print("  - Download ILSVRC2012_devkit_t12.tar.gz from ImageNet website")
            print("  - Extract it and find ILSVRC2012_validation_ground_truth.txt")
            print("  - Use: --ground-truth path/to/ILSVRC2012_validation_ground_truth.txt")
    
    print("\n" + "=" * 60)
    print("Setup Summary:")
    print(f"  Training data: {train_output} {'OK' if train_output.exists() and any(train_output.iterdir()) else 'X'}")
    print(f"  Validation data: {val_output} {'OK' if val_output.exists() and any(val_output.iterdir()) else 'X'}")
    print("=" * 60)
    
    if train_output.exists() and val_output.exists():
        print("\n[OK] Dataset is ready! You can now run training with:")
        print(f"  python main_when.py --data {args.data_dir}")
