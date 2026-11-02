"""
Create a FULLY CORRECTED ImageNet validation folder based on Google ReaL Labels,
while KEEPING uncertain samples in their original folders.

This script:
1. Keeps ILSVRC2012_img_val UNTOUCHED
2. Creates ILSVRC2012_img_val_Google_FULLY_CORRECTED_keep_uncertain
3. Moves mislabeled samples to their CORRECT folders (according to Google ReaL)
4. Keeps uncertain samples (empty labels) in their ORIGINAL folders

Difference from create_fully_corrected_val_Google_ReaL.py:
- That script: Skips uncertain samples -> 46,837 samples
- This script: Keeps uncertain in original folders -> 50,000 samples

Source: "Are we done with ImageNet?"
https://arxiv.org/abs/2006.07159
https://github.com/google-research/reassessed-imagenet

IMPORTANT: Google ReaL labels use PyTorch ImageFolder class indices (sorted WNIDs)
"""

import os
import shutil
import json
from tqdm import tqdm
from torchvision import datasets

print("=" * 80)
print("Create FULLY Corrected Google ReaL Folder (Keep Uncertain)")
print("=" * 80)
print()

# Configuration
SOURCE_DIR = "ILSVRC2012_img_val"
TARGET_DIR = "ILSVRC2012_img_val_Google_FULLY_CORRECTED_keep_uncertain"
REAL_LABELS_FILE = "Google_ReaL_labels.json"
REPORT_FILE = "Google_ReaL_fully_corrected_keep_uncertain_folder_report.txt"

# Load Google ReaL labels
print("Loading Google ReaL labels...")
if os.path.exists(REAL_LABELS_FILE):
    with open(REAL_LABELS_FILE, 'r') as f:
        real_labels = json.load(f)
    print(f"[OK] Loaded {len(real_labels)} ReaL label entries")
else:
    print(f"ERROR: Google ReaL labels file not found: {REAL_LABELS_FILE}")
    exit(1)

# Load ImageFolder dataset (FAST - pre-indexed)
print("Loading validation dataset with PyTorch ImageFolder...")
val_dataset = datasets.ImageFolder(SOURCE_DIR)
print(f"[OK] Loaded {len(val_dataset)} samples from {len(val_dataset.classes)} classes")

print()
print(f"Creating fully corrected (keep uncertain) folder: {TARGET_DIR}")
os.makedirs(TARGET_DIR, exist_ok=True)

# Create all class folders
print("Creating class folders...")
for class_name in val_dataset.classes:
    os.makedirs(os.path.join(TARGET_DIR, class_name), exist_ok=True)

# Process all validation files
copied_correctly_labeled = []  # Already in correct folder
corrected_mislabeled = []      # Moved to correct folder
kept_uncertain = []            # No valid labels -> kept in original folder
multi_label_samples = []       # Multiple valid labels

print()
print("Processing samples based on Google ReaL reassessment...")
print("Mislabeled -> correct folders; uncertain -> original folders")
print()

for idx, (img_path, original_folder_idx) in enumerate(tqdm(val_dataset.samples, desc="Processing")):
    filename = os.path.basename(img_path)

    # Extract file number from filename (ILSVRC2012_val_00000001.JPEG -> 1)
    file_num = int(filename.split('_')[-1].split('.')[0])

    # Get ReaL labels for this image using file number (1-indexed -> 0-indexed)
    reassessed_labels = real_labels[file_num - 1]

    if not isinstance(reassessed_labels, list):
        continue

    target_folder_idx = None

    if len(reassessed_labels) == 0:
        # No valid labels - uncertain -> KEEP in original folder
        target_folder_idx = original_folder_idx
        kept_uncertain.append({
            'index': idx,
            'filename': filename,
            'folder': val_dataset.classes[original_folder_idx],
            'folder_idx': original_folder_idx,
            'reason': "No valid labels (uncertain) - kept in original folder"
        })
    elif original_folder_idx in reassessed_labels:
        # Current folder is correct! Keep it there
        target_folder_idx = original_folder_idx
        copied_correctly_labeled.append({
            'index': idx,
            'filename': filename,
            'folder': val_dataset.classes[original_folder_idx],
            'folder_idx': original_folder_idx
        })
    else:
        # Current folder is WRONG! Move to correct folder (use first valid label)
        target_folder_idx = reassessed_labels[0]
        corrected_mislabeled.append({
            'index': idx,
            'filename': filename,
            'old_folder': val_dataset.classes[original_folder_idx],
            'old_folder_idx': original_folder_idx,
            'new_folder': val_dataset.classes[target_folder_idx],
            'new_folder_idx': target_folder_idx,
            'all_valid_labels': reassessed_labels
        })

    # Track multi-label samples
    if len(reassessed_labels) > 1:
        multi_label_samples.append({
            'index': idx,
            'filename': filename,
            'chosen_label': target_folder_idx,
            'all_labels': reassessed_labels
        })

    # Copy file to target folder
    target_folder_name = val_dataset.classes[target_folder_idx]
    target_path = os.path.join(TARGET_DIR, target_folder_name, filename)
    shutil.copy2(img_path, target_path)

num_total = len(val_dataset)
num_correctly_labeled = len(copied_correctly_labeled)
num_corrected = len(corrected_mislabeled)
num_uncertain = len(kept_uncertain)
num_copied_total = num_correctly_labeled + num_corrected + num_uncertain
num_multi_label = len(multi_label_samples)

print()
print("[OK] Fully corrected (keep uncertain) folder created")
print()

# Generate report
print(f"Writing report to: {REPORT_FILE}")
with open(REPORT_FILE, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("Google ReaL Labels FULLY CORRECTED (Keep Uncertain) Folder Report\n")
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
    f.write(f"Original (UNTOUCHED):     {SOURCE_DIR} ({num_total} samples)\n")
    f.write(f"Fully Corrected+Keep (NEW): {TARGET_DIR} ({num_copied_total} samples)\n")
    f.write(f"  - Already correct:      {num_correctly_labeled} samples (kept in same folder)\n")
    f.write(f"  - Corrected:            {num_corrected} samples (moved to correct folder)\n")
    f.write(f"  - Uncertain (kept):     {num_uncertain} samples (kept in original folder)\n\n")

    f.write("STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total original samples:         {num_total:,}\n")
    f.write(f"Copied (all samples):           {num_copied_total:,} ({num_copied_total/num_total*100:.2f}%)\n")
    f.write(f"  - Already correctly labeled:  {num_correctly_labeled:,} ({num_correctly_labeled/num_total*100:.2f}%)\n")
    f.write(f"  - Fixed (moved to correct):   {num_corrected:,} ({num_corrected/num_total*100:.2f}%)\n")
    f.write(f"  - Uncertain (kept original):  {num_uncertain:,} ({num_uncertain/num_total*100:.2f}%)\n")
    f.write(f"Multi-label samples:            {num_multi_label:,} (first valid label used)\n\n")

    if num_corrected > 0:
        f.write("CORRECTED SAMPLES (First 100)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'Filename':<30} {'Old Folder':<15} {'New Folder':<15}\n")
        f.write("-" * 80 + "\n")
        for sample in corrected_mislabeled[:100]:
            f.write(f"{sample['index']:<8} {sample['filename']:<30} {sample['old_folder']:<15} {sample['new_folder']:<15}\n")
        if num_corrected > 100:
            f.write(f"\n... and {num_corrected - 100} more\n")

    if num_uncertain > 0:
        f.write("\nKEPT UNCERTAIN SAMPLES (First 100)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'Filename':<30} {'Folder':<15}\n")
        f.write("-" * 80 + "\n")
        for sample in kept_uncertain[:100]:
            f.write(f"{sample['index']:<8} {sample['filename']:<30} {sample['folder']:<15}\n")
        if num_uncertain > 100:
            f.write(f"\n... and {num_uncertain - 100} more\n")

    if num_multi_label > 0:
        f.write("\nMULTI-LABEL SAMPLES (First 50)\n")
        f.write("-" * 80 + "\n")
        f.write("These samples have multiple valid labels. First valid label was used.\n")
        f.write(f"{'Index':<8} {'Filename':<30} {'Chosen':<8} {'All Valid':<20}\n")
        f.write("-" * 80 + "\n")
        for sample in multi_label_samples[:50]:
            all_labels_str = str(sample['all_labels'])[:20]
            f.write(f"{sample['index']:<8} {sample['filename']:<30} {sample['chosen_label']:<8} {all_labels_str:<20}\n")
        if num_multi_label > 50:
            f.write(f"\n... and {num_multi_label - 50} more\n")

    f.write("\nNOTES\n")
    f.write("-" * 80 + "\n")
    f.write("- Original folder remains UNTOUCHED\n")
    f.write("- Mislabeled samples were MOVED to their correct folders\n")
    f.write("- Uncertain samples (empty labels) were KEPT in original folders\n")
    f.write("- Multi-label images: first valid label used (or current if valid)\n")
    f.write("- Expected total: 50,000 samples (full validation set)\n")
    f.write("- ReaL labels use PyTorch ImageFolder ordering (sorted WNIDs)\n")

print("[OK] Report saved")
print()

# Print summary
print("=" * 80)
print("FULLY CORRECTED (KEEP UNCERTAIN) FOLDER CREATED")
print("=" * 80)
print(f"Original folder (untouched):  {SOURCE_DIR} ({num_total} samples)")
print(f"New folder:                   {TARGET_DIR} ({num_copied_total} samples)")
print()
print(f"Already correctly labeled:    {num_correctly_labeled} samples (kept)")
print(f"Fixed (moved to correct):     {num_corrected} samples (corrected!)")
print(f"Uncertain (kept original):    {num_uncertain} samples")
print(f"Multi-label samples:          {num_multi_label} samples")
print()
print(f"Report saved to: {REPORT_FILE}")
print()
print(f"  valdir = '{TARGET_DIR}'")
print("=" * 80)
