"""
Universal script to move mislabeled samples identified by any detection method
to a separate directory (ILSVRC2012_img_valWRONG).

Works with reports from:
- detect_mislabeled_samples.py (cleanlab with your model)
- detect_mislabeled_MIT_cleanlab.py (MIT official list)
- detect_mislabeled_Google_ReaL.py (Google ReaL labels)

Usage: Update REPORT_JSON variable to point to your desired report file.
"""

import os
import json
import shutil
import argparse
from tqdm import tqdm

print("=" * 80)
print("Universal Mislabeled Samples Mover")
print("=" * 80)
print()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Move mislabeled ImageNet validation samples')
parser.add_argument('--report', type=str, 
                    default='mislabeled_samples_report.json',
                    help='Path to the mislabeled samples report JSON file')
parser.add_argument('--source', type=str,
                    default='ILSVRC2012_img_val',
                    help='Source validation directory')
parser.add_argument('--target', type=str,
                    default='ILSVRC2012_img_valWRONG',
                    help='Target directory for mislabeled samples')
args = parser.parse_args()

REPORT_JSON = args.report
SOURCE_DIR = args.source
TARGET_DIR = args.target
MOVE_LOG = f"move_mislabeled_log_{os.path.basename(REPORT_JSON).replace('.json', '')}.txt"

# Check if report exists
if not os.path.exists(REPORT_JSON):
    print(f"ERROR: Report file not found: {REPORT_JSON}")
    print()
    print("Available report files:")
    for filename in os.listdir('.'):
        if filename.startswith('mislabeled_samples') and filename.endswith('.json'):
            print(f"  - {filename}")
    print()
    print("Usage: python move_mislabeled_universal.py --report <report_file>")
    exit(1)

# Load report
print(f"Loading report from: {REPORT_JSON}")
with open(REPORT_JSON, 'r') as f:
    report = json.load(f)

summary = report['summary']
mislabeled_samples = report['mislabeled_samples']

print(f"✓ Report loaded")
print(f"  Source:                       {summary.get('source', 'Unknown')}")
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
print(f"Report:      {REPORT_JSON}")
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
    log_file.write(f"Report file:      {REPORT_JSON}\n")
    log_file.write(f"Source:           {summary.get('source', 'Unknown')}\n")
    log_file.write(f"Source directory: {SOURCE_DIR}\n")
    log_file.write(f"Target directory: {TARGET_DIR}\n")
    log_file.write(f"Total files:      {len(mislabeled_samples)}\n\n")
    
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
                'relative_path': relative_path
            })
            
            log_file.write(f"MOVED: {relative_path}\n")
            
            # Write additional info based on report type
            if 'label_quality_score' in sample:
                log_file.write(f"  Quality Score: {sample['label_quality_score']:.4f}\n")
            if 'given_class' in sample:
                log_file.write(f"  Given Class: {sample['given_class']}\n")
            if 'predicted_class' in sample:
                log_file.write(f"  Predicted Class: {sample['predicted_class']}\n")
            if 'original_class' in sample:
                log_file.write(f"  Original Class: {sample['original_class']}\n")
            if 'reassessed_class' in sample:
                log_file.write(f"  Reassessed Class: {sample['reassessed_class']}\n")
            if 'reassessed_classes' in sample:
                log_file.write(f"  Reassessed Classes: {', '.join(sample['reassessed_classes'])}\n")
            
            log_file.write("\n")
            
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
            remaining_count += len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
print(f"  {remaining_count:,} samples")
print()

print(f"Mislabeled samples in target directory:")
wrong_count = 0
if os.path.exists(TARGET_DIR):
    for class_dir in os.listdir(TARGET_DIR):
        class_path = os.path.join(TARGET_DIR, class_dir)
        if os.path.isdir(class_path):
            wrong_count += len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
print(f"  {wrong_count:,} samples")
print()

print("=" * 80)

