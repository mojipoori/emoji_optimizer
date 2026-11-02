"""
Create ImageNet validation folder with ONLY uncertain samples removed
Based on Google ReaL Labels

This script:
- Removes uncertain samples (empty labels in Google ReaL)
- Keeps all other samples in their ORIGINAL folders (even if mislabeled)
- Does NOT move mislabeled samples to correct folders
- Does NOT handle multi-label samples specially

Original folder: ILSVRC2012_img_val (untouched)
Output folder: ILSVRC2012_img_val_Google_only_uncertain_cleaned
"""

import os
import shutil
import json
from tqdm import tqdm
from torchvision import datasets

# Configuration
SOURCE_DIR = "ILSVRC2012_img_val"
TARGET_DIR = "ILSVRC2012_img_val_Google_only_uncertain_cleaned"
REAL_LABELS_FILE = "Google_ReaL_labels.json"
REPORT_FILE = "Google_ReaL_only_uncertain_cleaned_folder_report.txt"

print("=" * 80)
print("Creating Google ReaL Validation Folder (Only Uncertain Cleaned)")
print("=" * 80)
print()
print(f"Source directory: {SOURCE_DIR}")
print(f"Target directory: {TARGET_DIR}")
print(f"ReaL labels file: {REAL_LABELS_FILE}")
print()

# Load Google ReaL labels
print("Loading Google ReaL labels...")
with open(REAL_LABELS_FILE, 'r') as f:
    real_labels = json.load(f)

print(f"Loaded {len(real_labels)} ReaL label entries")
print()

# Load validation dataset with PyTorch ImageFolder
print("Loading validation dataset...")
val_dataset = datasets.ImageFolder(SOURCE_DIR)
print(f"Found {len(val_dataset.samples)} images in {len(val_dataset.classes)} classes")
print()

# Create target directory structure
print("Creating target directory structure...")
os.makedirs(TARGET_DIR, exist_ok=True)

# Create all class subfolders
for class_name in val_dataset.classes:
    os.makedirs(os.path.join(TARGET_DIR, class_name), exist_ok=True)

print(f"Created {len(val_dataset.classes)} class folders")
print()

# Process samples
print("Processing samples...")
print()

copied_samples = []
skipped_uncertain = []

for idx, (img_path, original_folder_idx) in enumerate(tqdm(val_dataset.samples, desc="Processing")):
    filename = os.path.basename(img_path)
    
    # Extract the 1-indexed image number from the filename
    # e.g., ILSVRC2012_val_00000001.JPEG -> 1
    image_number = int(filename.split('_')[-1].split('.')[0])
    
    # Get the corresponding ReaL labels entry (0-indexed list)
    real_label_entry = real_labels[image_number - 1]
    
    if len(real_label_entry) == 0:
        # No valid labels - uncertain/problematic - SKIP
        skipped_uncertain.append({
            'index': idx,
            'filename': filename,
            'original_folder': val_dataset.classes[original_folder_idx]
        })
    else:
        # Has at least one valid label - COPY to original folder
        # We don't care if it's mislabeled or multi-label
        # Just keep it in its current folder
        original_folder_name = val_dataset.classes[original_folder_idx]
        
        source_path = img_path
        target_path = os.path.join(TARGET_DIR, original_folder_name, filename)
        
        shutil.copy2(source_path, target_path)
        
        copied_samples.append({
            'index': idx,
            'filename': filename,
            'folder': original_folder_name,
            'num_real_labels': len(real_label_entry)
        })

print()
print("=" * 80)
print("Processing Complete!")
print("=" * 80)
print()

# Generate detailed report
print("Generating report...")

total_samples = len(val_dataset.samples)
num_copied = len(copied_samples)
num_uncertain = len(skipped_uncertain)

report_lines = []
report_lines.append("=" * 80)
report_lines.append("Google ReaL Labels - Only Uncertain Cleaned Folder Report")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("Configuration:")
report_lines.append(f"  Source directory: {SOURCE_DIR}")
report_lines.append(f"  Target directory: {TARGET_DIR}")
report_lines.append(f"  ReaL labels file: {REAL_LABELS_FILE}")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("Summary Statistics")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append(f"Total samples in source:          {total_samples:,}")
report_lines.append(f"Samples copied (kept):            {num_copied:,} ({num_copied/total_samples*100:.2f}%)")
report_lines.append(f"Samples skipped (uncertain):      {num_uncertain:,} ({num_uncertain/total_samples*100:.2f}%)")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("Important Notes")
report_lines.append("=" * 80)
report_lines.append("")
report_lines.append("1. This folder contains ALL samples EXCEPT uncertain ones")
report_lines.append("2. Mislabeled samples are KEPT in their ORIGINAL (wrong) folders")
report_lines.append("3. Multi-label samples are KEPT in their current folders")
report_lines.append("4. Only samples with empty ReaL labels [] are removed")
report_lines.append("")
report_lines.append("This is different from:")
report_lines.append("  - 'CORRECTED' folder: only keeps correctly labeled samples")
report_lines.append("  - 'FULLY_CORRECTED' folder: moves mislabeled to correct folders")
report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("Breakdown by Label Type")
report_lines.append("=" * 80)
report_lines.append("")

# Count by label type
single_label = sum(1 for s in copied_samples if s['num_real_labels'] == 1)
multi_label = sum(1 for s in copied_samples if s['num_real_labels'] > 1)

report_lines.append(f"Single-label samples (1 valid label):  {single_label:,} ({single_label/num_copied*100:.2f}%)")
report_lines.append(f"Multi-label samples (>1 valid label):  {multi_label:,} ({multi_label/num_copied*100:.2f}%)")
report_lines.append("")
report_lines.append("Note: Both single and multi-label samples kept in original folders")
report_lines.append("      regardless of whether they are correctly or incorrectly labeled")
report_lines.append("")

# Save report
report_content = "\n".join(report_lines)

with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(report_content)
print()
print(f"Report saved to: {REPORT_FILE}")
print()

# Additional statistics
print("=" * 80)
print("Additional Statistics")
print("=" * 80)
print()
print(f"Average labels per copied sample: {sum(s['num_real_labels'] for s in copied_samples) / num_copied:.2f}")
print()

# Sample some uncertain files
print("=" * 80)
print("Sample Uncertain Files (first 10)")
print("=" * 80)
print()
for i, sample in enumerate(skipped_uncertain[:10]):
    print(f"{i+1}. {sample['filename']} (was in folder: {sample['original_folder']})")
print()

print("=" * 80)
print("Done!")
print("=" * 80)
print()
print(f"Created folder: {TARGET_DIR}")
print(f"Contains: {num_copied:,} samples")
print(f"Removed: {num_uncertain:,} uncertain samples")
print()

