"""
Simple setup script to organize ImageNet dataset
This script will extract training data and help organize validation data
"""

import os
import sys
from pathlib import Path
from prepare_imagenet_data import (
    extract_tar_files, 
    organize_validation_with_ground_truth,
    organize_validation_with_mapping,
    get_class_names_from_tar_dir
)

def main():
    print("=" * 70)
    print("ImageNet Dataset Setup")
    print("=" * 70)
    print("\nThis script will organize your ImageNet data to match the code structure.")
    print("Expected structure after setup:")
    print("  ./train/")
    print("    class1/")
    print("    class2/")
    print("    ...")
    print("  ./val/")
    print("    class1/")
    print("    class2/")
    print("    ...")
    print("\n" + "=" * 70)
    
    # Check if directories exist
    train_tar_dir = Path("ILSVRC2012_img_train")
    val_dir = Path("ILSVRC2012_img_val")
    train_output = Path("train")
    val_output = Path("val")
    
    if not train_tar_dir.exists():
        print(f"\n❌ Error: Training directory not found: {train_tar_dir}")
        print("   Please make sure ILSVRC2012_img_train/ exists in the current directory")
        return False
    
    if not val_dir.exists():
        print(f"\n❌ Error: Validation directory not found: {val_dir}")
        print("   Please make sure ILSVRC2012_img_val/ exists in the current directory")
        return False
    
    # Step 1: Extract training data
    print("\n[Step 1/2] Extracting training data...")
    if train_output.exists() and any(train_output.iterdir()):
        response = input(f"  Directory {train_output} already exists. Re-extract? (y/N): ")
        if response.lower() != 'y':
            print(f"  ✓ Skipping training extraction (using existing {train_output})")
        else:
            print(f"  Extracting {len(list(train_tar_dir.glob('*.tar')))} tar files...")
            extract_tar_files(str(train_tar_dir), str(train_output))
    else:
        print(f"  Extracting {len(list(train_tar_dir.glob('*.tar')))} tar files...")
        extract_tar_files(str(train_tar_dir), str(train_output))
    
    # Step 2: Organize validation data
    print("\n[Step 2/2] Organizing validation data...")
    if val_output.exists() and any(val_output.iterdir()):
        response = input(f"  Directory {val_output} already exists. Re-organize? (y/N): ")
        if response.lower() != 'y':
            print(f"  ✓ Skipping validation organization (using existing {val_output})")
            return True
    
    # Check for ground truth file
    ground_truth_file = None
    possible_gt_files = [
        "ILSVRC2012_validation_ground_truth.txt",
        "val_ground_truth.txt",
        "validation_ground_truth.txt"
    ]
    
    for gt_file in possible_gt_files:
        if Path(gt_file).exists():
            ground_truth_file = gt_file
            break
    
    if ground_truth_file:
        print(f"  Found ground truth file: {ground_truth_file}")
        class_names = get_class_names_from_tar_dir(train_tar_dir)
        organize_validation_with_ground_truth(str(val_dir), str(val_output), ground_truth_file, class_names)
    else:
        # Check for mapping file
        mapping_file = None
        possible_mapping_files = [
            "val_mapping.txt",
            "validation_mapping.txt"
        ]
        
        for map_file in possible_mapping_files:
            if Path(map_file).exists():
                mapping_file = map_file
                break
        
        if mapping_file:
            print(f"  Found mapping file: {mapping_file}")
            organize_validation_with_mapping(str(val_dir), str(val_output), mapping_file)
        else:
            print("\n  ⚠ Warning: No ground truth or mapping file found!")
            print("\n  To organize validation data, you need one of:")
            print("    1. ILSVRC2012_validation_ground_truth.txt")
            print("       Download from: ILSVRC2012_devkit_t12.tar.gz")
            print("       Then run: python prepare_imagenet_data.py --organize-val --ground-truth ILSVRC2012_validation_ground_truth.txt")
            print("\n    2. val_mapping.txt (image_name class_name format)")
            print("       Then run: python prepare_imagenet_data.py --organize-val --val-mapping val_mapping.txt")
            print("\n  For now, validation data will not be organized.")
            print("  You can organize it later using prepare_imagenet_data.py")
            return False
    
    print("\n" + "=" * 70)
    print("✓ Setup Complete!")
    print("=" * 70)
    print(f"\nYour dataset is now organized:")
    print(f"  Training: {train_output.absolute()}")
    print(f"  Validation: {val_output.absolute()}")
    print(f"\nYou can now run training with:")
    print(f"  python main_when.py --data .")
    print("=" * 70)
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

