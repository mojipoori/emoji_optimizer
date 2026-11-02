"""
Script to move mislabeled samples identified by detect_mislabeled_samples.py
to a separate directory (ILSVRC2012_img_valWRONG).

This script reads the mislabeled_samples_report.json file and moves (cuts) the
identified mislabeled samples from ILSVRC2012_img_val to ILSVRC2012_img_valWRONG,
maintaining the class folder structure.
"""

import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

print("=" * 80)
print("Move Mislabeled Samples to Separate Directory")
print("=" * 80)
print()

# Configuration
REPORT_JSON = "mislabeled_samples_report.json"
SOURCE_DIR = "ILSVRC2012_img_val"
TARGET_DIR = "ILSVRC2012_img_valWRONG"
MOVE_LOG = "move_mislabeled_log.txt"

# Check if report exists
if not os.path.exists(REPORT_JSON):
    print(f"ERROR: Report file not found: {REPORT_JSON}")
    print("Please run 'detect_mislabeled_samples.py' first to generate the report.")
    exit(1)

# Load report
print(f"Loading report from: {REPORT_JSON}")
with open(REPORT_JSON, 'r') as f:
    report = json.load(f)

summary = report['summary']
mislabeled_samples = report['mislabeled_samples']

print(f"✓ Report loaded")
print(f"  Total samples in dataset:     {summary['total_samples']:,}")
print(f"  Mislabeled samples to move:   {summary['num_mislabeled']:,}")
print()

# Confirm action
print("=" * 80)
print("WARNING: This operation will MOVE (cut) files from the source directory!")
print("=" * 80)
print(f"Source:      {SOURCE_DIR}")
print(f"Destination: {TARGET_DIR}")
print(f"Files:       {len(mislabeled_samples):,} samples")
print()

response = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
if response != 'yes':
    print("Operation cancelled by user.")
    exit(0)

print()
print("Starting file move operation...")

# Create target directory structure
print(f"Creating target directory: {TARGET_DIR}")
os.makedirs(TARGET_DIR, exist_ok=True)

# Track moved files
moved_files = []
failed_moves = []
skipped_files = []

# Move files
with open(MOVE_LOG, 'w', encoding='utf-8') as log_file:
    log_file.write("=" * 80 + "\n")
    log_file.write("Mislabeled Samples Move Log\n")
    log_file.write("=" * 80 + "\n\n")
    log_file.write(f"Source directory: {SOURCE_DIR}\n")
    log_file.write(f"Target directory: {TARGET_DIR}\n")
    log_file.write(f"Total files to move: {len(mislabeled_samples)}\n\n")
    
    for sample in tqdm(mislabeled_samples, desc="Moving files"):
        source_path = sample['file_path']
        relative_path = sample['relative_path']
        
        # Check if source file exists
        if not os.path.exists(source_path):
            skipped_files.append({
                'path': source_path,
                'reason': 'File not found'
            })
            log_file.write(f"SKIPPED: {relative_path} - File not found\n")
            continue
        
        # Construct target path
        target_path = os.path.join(TARGET_DIR, relative_path)
        
        # Create target class directory if needed
        target_class_dir = os.path.dirname(target_path)
        os.makedirs(target_class_dir, exist_ok=True)
        
        try:
            # Move file
            shutil.move(source_path, target_path)
            
            moved_files.append({
                'source': source_path,
                'target': target_path,
                'relative_path': relative_path,
                'quality_score': sample['label_quality_score'],
                'given_class': sample['given_class'],
                'predicted_class': sample['predicted_class']
            })
            
            log_file.write(f"MOVED: {relative_path}\n")
            log_file.write(f"  Quality Score: {sample['label_quality_score']:.4f}\n")
            log_file.write(f"  Given Class: {sample['given_class']}\n")
            log_file.write(f"  Predicted Class: {sample['predicted_class']}\n\n")
            
        except Exception as e:
            failed_moves.append({
                'path': source_path,
                'error': str(e)
            })
            log_file.write(f"FAILED: {relative_path} - {str(e)}\n")
    
    # Write summary to log
    log_file.write("\n" + "=" * 80 + "\n")
    log_file.write("SUMMARY\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"Successfully moved:  {len(moved_files)}\n")
    log_file.write(f"Failed:              {len(failed_moves)}\n")
    log_file.write(f"Skipped:             {len(skipped_files)}\n")
    log_file.write(f"Total:               {len(mislabeled_samples)}\n")

# Clean up empty class directories in source
print("\nCleaning up empty directories in source...")
for class_dir in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_dir)
    if os.path.isdir(class_path) and not os.listdir(class_path):
        os.rmdir(class_path)
        print(f"  Removed empty directory: {class_dir}")

print()
print("=" * 80)
print("OPERATION COMPLETED")
print("=" * 80)
print(f"Successfully moved:      {len(moved_files):,} files")
print(f"Failed:                  {len(failed_moves):,} files")
print(f"Skipped:                 {len(skipped_files):,} files")
print()
print(f"Target directory:        {TARGET_DIR}")
print(f"Move log saved to:       {MOVE_LOG}")
print()

if failed_moves:
    print("WARNING: Some files failed to move. Check the log file for details.")
    print("Failed files:")
    for fail in failed_moves[:10]:
        print(f"  - {fail['path']}: {fail['error']}")
    if len(failed_moves) > 10:
        print(f"  ... and {len(failed_moves) - 10} more (see log file)")
    print()

if skipped_files:
    print(f"Note: {len(skipped_files)} files were skipped (not found). Check log for details.")
    print()

print("Remaining samples in source directory:")
remaining_count = 0
if os.path.exists(SOURCE_DIR):
    for class_dir in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_dir)
        if os.path.isdir(class_path):
            remaining_count += len(os.listdir(class_path))
print(f"  {remaining_count:,} samples")
print()

print(f"Mislabeled samples in target directory:")
wrong_count = 0
if os.path.exists(TARGET_DIR):
    for class_dir in os.listdir(TARGET_DIR):
        class_path = os.path.join(TARGET_DIR, class_dir)
        if os.path.isdir(class_path):
            wrong_count += len(os.listdir(class_path))
print(f"  {wrong_count:,} samples")
print()

print("=" * 80)

