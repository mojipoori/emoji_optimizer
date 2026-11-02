"""
Create a CORRECTED ImageNet validation folder based on Google ReaL Labels.
This script:
1. Keeps ILSVRC2012_img_val UNTOUCHED
2. Creates ILSVRC2012_img_val_Google_CORRECTED with only correctly labeled samples
3. Uses the official Google ReaL reassessed labels

Source: "Are we done with ImageNet?"
https://arxiv.org/abs/2006.07159
https://github.com/google-research/reassessed-imagenet

IMPORTANT: Google ReaL labels use PyTorch ImageFolder class indices (sorted WNIDs)
"""

import os
import shutil
import json
from tqdm import tqdm

print("=" * 80)
print("Create Google ReaL Corrected Validation Folder")
print("=" * 80)
print()

# Configuration
SOURCE_DIR = "ILSVRC2012_img_val"
TARGET_DIR = "ILSVRC2012_img_val_Google_CORRECTED"
REAL_LABELS_FILE = "Google_ReaL_labels.json"
REPORT_FILE = "Google_ReaL_corrected_folder_report.txt"

# Load Google ReaL labels
print("Loading Google ReaL labels...")
if os.path.exists(REAL_LABELS_FILE):
    with open(REAL_LABELS_FILE, 'r') as f:
        real_labels = json.load(f)
    print(f"[OK] Loaded {len(real_labels)} ReaL label entries")
else:
    print(f"ERROR: Google ReaL labels file not found: {REAL_LABELS_FILE}")
    exit(1)

# Get sorted folder list (PyTorch ImageFolder ordering)
print("Scanning validation folder structure...")
folders = sorted([f for f in os.listdir(SOURCE_DIR) 
                  if os.path.isdir(os.path.join(SOURCE_DIR, f))])
print(f"[OK] Found {len(folders)} class folders")

# Create folder_name to index mapping
folder_to_idx = {folder: idx for idx, folder in enumerate(folders)}

print()
print(f"Creating corrected validation folder: {TARGET_DIR}")
os.makedirs(TARGET_DIR, exist_ok=True)

# Process all validation files in order
copied_samples = []
skipped_mislabeled = []
skipped_uncertain = []
multi_label_samples = []

print("Processing samples based on Google ReaL reassessment...")
print("(ReaL labels use PyTorch ImageFolder class indices - sorted WNIDs)")
print()

# Iterate through files in order (ILSVRC2012_val_00000001.JPEG to ILSVRC2012_val_00050000.JPEG)
for img_idx in tqdm(range(len(real_labels)), desc="Processing"):
    filename = f"ILSVRC2012_val_{img_idx + 1:08d}.JPEG"
    
    # Find which folder contains this file
    found_folder = None
    for folder in folders:
        folder_path = os.path.join(SOURCE_DIR, folder)
        file_path = os.path.join(folder_path, filename)
        if os.path.exists(file_path):
            found_folder = folder
            break
    
    if not found_folder:
        print(f"\nWarning: File {filename} not found in any folder!")
        continue
    
    # Get folder index (this is the "class" that models predict)
    folder_idx = folder_to_idx[found_folder]
    
    # Get ReaL labels for this image
    reassessed_labels = real_labels[img_idx]
    
    # Check if sample should be kept
    should_copy = False
    reason = ""
    
    if isinstance(reassessed_labels, list):
        if len(reassessed_labels) == 0:
            # No valid labels - uncertain/problematic
            reason = "No valid labels (uncertain)"
            skipped_uncertain.append({
                'index': img_idx,
                'filename': filename,
                'folder': found_folder,
                'folder_idx': folder_idx,
                'reason': reason
            })
        elif folder_idx in reassessed_labels:
            # Current folder label is correct!
            should_copy = True
            if len(reassessed_labels) > 1:
                multi_label_samples.append(img_idx)
        else:
            # Current folder label is WRONG
            correct_folders = [folders[idx] for idx in reassessed_labels if idx < len(folders)]
            reason = f"Mislabeled (should be {correct_folders})"
            skipped_mislabeled.append({
                'index': img_idx,
                'filename': filename,
                'folder': found_folder,
                'folder_idx': folder_idx,
                'reassessed_labels': reassessed_labels,
                'correct_folders': correct_folders,
                'reason': reason
            })
    
    # Copy if should be kept
    if should_copy:
        source_path = os.path.join(SOURCE_DIR, found_folder, filename)
        target_class_dir = os.path.join(TARGET_DIR, found_folder)
        target_path = os.path.join(target_class_dir, filename)
        
        # Create class directory
        os.makedirs(target_class_dir, exist_ok=True)
        
        # Copy file
        shutil.copy2(source_path, target_path)
        copied_samples.append({
            'index': img_idx,
            'filename': filename,
            'folder': found_folder
        })

num_total = len(real_labels)
num_copied = len(copied_samples)
num_mislabeled = len(skipped_mislabeled)
num_uncertain = len(skipped_uncertain)
num_multi_label = len(multi_label_samples)
total_skipped = num_mislabeled + num_uncertain

print()
print("[OK] Corrected folder created")
print()

# Generate report
print(f"Writing report to: {REPORT_FILE}")
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Google ReaL Labels Corrected Validation Folder Report\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("SOURCE\n")
    f.write("-" * 80 + "\n")
    f.write("Paper: Are we done with ImageNet?\n")
    f.write("Authors: Lucas Beyer, Olivier J. Henaff, Alexander Kolesnikov, et al.\n")
    f.write("Organization: Google Research\n")
    f.write("Published: 2020\n")
    f.write("URL: https://arxiv.org/abs/2006.07159\n")
    f.write("GitHub: https://github.com/google-research/reassessed-imagenet\n\n")
    
    f.write("FOLDERS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Original (UNTOUCHED): {SOURCE_DIR} ({num_total} samples)\n")
    f.write(f"Corrected (NEW):      {TARGET_DIR} ({num_copied} samples)\n")
    f.write(f"Removed samples:      {total_skipped} total\n")
    f.write(f"  - Mislabeled:       {num_mislabeled}\n")
    f.write(f"  - Uncertain:        {num_uncertain}\n\n")
    
    f.write("STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total original samples:     {num_total:,}\n")
    f.write(f"Copied (clean) samples:     {num_copied:,} ({num_copied/num_total*100:.2f}%)\n")
    f.write(f"Skipped (mislabeled):       {num_mislabeled:,} ({num_mislabeled/num_total*100:.2f}%)\n")
    f.write(f"Skipped (uncertain):        {num_uncertain:,} ({num_uncertain/num_total*100:.2f}%)\n")
    f.write(f"Multi-label samples:        {num_multi_label:,} (kept, as current label is valid)\n\n")
    
    if num_mislabeled > 0:
        f.write("SKIPPED MISLABELED SAMPLES (First 100)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'Filename':<30} {'Current':<15} {'Should Be':<15}\n")
        f.write("-" * 80 + "\n")
        for sample in skipped_mislabeled[:100]:
            correct = sample['correct_folders'][0] if sample['correct_folders'] else 'unknown'
            f.write(f"{sample['index']:<8} {sample['filename']:<30} {sample['folder']:<15} {correct:<15}\n")
        if num_mislabeled > 100:
            f.write(f"\n... and {num_mislabeled - 100} more\n")
    
    if num_uncertain > 0:
        f.write("\nSKIPPED UNCERTAIN SAMPLES (First 100)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'Filename':<30} {'Folder':<15}\n")
        f.write("-" * 80 + "\n")
        for sample in skipped_uncertain[:100]:
            f.write(f"{sample['index']:<8} {sample['filename']:<30} {sample['folder']:<15}\n")
        if num_uncertain > 100:
            f.write(f"\n... and {num_uncertain - 100} more\n")
    
    f.write("\nNOTES\n")
    f.write("-" * 80 + "\n")
    f.write("- Original folder remains UNTOUCHED\n")
    f.write("- Corrected folder contains only correctly labeled samples\n")
    f.write("- Multi-label images are kept if current label is among valid labels\n")
    f.write("- Labels reassessed by multiple expert annotators\n")
    f.write("- Use corrected folder for more reliable benchmarking\n")
    f.write("- Google ReaL provides highest quality reassessment available\n")
    f.write("- ReaL labels use PyTorch ImageFolder ordering (sorted WNIDs)\n")

print("[OK] Report saved")
print()

# Print summary
print("=" * 80)
print("CORRECTED FOLDER CREATED")
print("=" * 80)
print(f"Original folder (untouched):  {SOURCE_DIR} ({num_total} samples)")
print(f"Corrected folder (new):       {TARGET_DIR} ({num_copied} samples)")
print(f"Removed samples:              {total_skipped} total")
print(f"  - Mislabeled:               {num_mislabeled} samples")
print(f"  - Uncertain:                {num_uncertain} samples")
print(f"Multi-label (kept):           {num_multi_label} samples")
print()
print(f"Clean samples:   {num_copied/num_total*100:.2f}%")
print(f"Removed:         {total_skipped/num_total*100:.2f}%")
print()
print(f"Report saved to: {REPORT_FILE}")
print()
print("You can now use the corrected folder for training/evaluation:")
print(f"  valdir = '{TARGET_DIR}'")
print("=" * 80)
