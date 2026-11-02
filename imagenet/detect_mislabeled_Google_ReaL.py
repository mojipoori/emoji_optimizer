"""
Script to identify mislabeled ImageNet validation samples using Google's ReaL Labels.
This uses the definitive reassessed labels from the research paper:
"Are we done with ImageNet?"
https://arxiv.org/abs/1902.10811
https://github.com/google-research/reassessed-imagenet

This is INDEPENDENT of your model - uses official research findings.
"""

import os
import json
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path

print("=" * 80)
print("ImageNet Validation - Google ReaL (Reassessed Labels) Official List")
print("=" * 80)
print()

# Configuration
VAL_DIR = "ILSVRC2012_img_val"
OUTPUT_JSON = "mislabeled_samples_Google_ReaL.json"
OUTPUT_TXT = "mislabeled_samples_Google_ReaL.txt"

# Google ReaL Labels URLs
# This contains reassessed labels for ImageNet validation set
REAL_LABELS_URL = "https://github.com/google-research/reassessed-imagenet/raw/master/real.json"

print("Downloading Google ReaL Labels official reassessed annotations...")
print(f"Source: {REAL_LABELS_URL}")
print()

# Download the ReaL labels
real_labels_data = None
try:
    response = requests.get(REAL_LABELS_URL, timeout=30)
    if response.status_code == 200:
        real_labels_data = response.json()
        with open("real_labels.json", "w") as f:
            json.dump(real_labels_data, f)
        print("✓ Downloaded Google ReaL labels")
    else:
        print(f"Warning: Could not download from GitHub (status {response.status_code})")
        # Try to load from local file if exists
        if os.path.exists("real_labels.json"):
            with open("real_labels.json", "r") as f:
                real_labels_data = json.load(f)
            print("✓ Loaded from local cache")
except Exception as e:
    print(f"Warning: Could not download: {e}")
    # Try to load from local file if exists
    if os.path.exists("real_labels.json"):
        with open("real_labels.json", "r") as f:
            real_labels_data = json.load(f)
        print("✓ Loaded from local cache")

print()

# Get validation dataset structure
print(f"Loading validation dataset structure from: {VAL_DIR}")

from torchvision import datasets
val_dataset = datasets.ImageFolder(VAL_DIR)

num_samples = len(val_dataset)
num_classes = len(val_dataset.classes)
print(f"✓ Dataset loaded: {num_samples} samples, {num_classes} classes")
print()

# Get all file paths and labels
sample_paths = [path for path, _ in val_dataset.samples]
sample_labels = np.array([label for _, label in val_dataset.samples])

print(f"Analyzing labels using Google ReaL reassessment...")
print()

# Create list of mislabeled samples
mislabeled_samples = []
multi_label_samples = []
uncertain_samples = []

if real_labels_data:
    # Parse ReaL labels data
    # The format varies - typically maps image index to reassessed label(s)
    
    print("Processing Google ReaL reassessed labels...")
    
    # ReaL labels format: index -> list of correct labels
    # If original label is not in the reassessed list, it's mislabeled
    
    for idx in tqdm(range(len(sample_paths)), desc="Checking labels"):
        str_idx = str(idx)
        
        if str_idx in real_labels_data:
            reassessed_labels = real_labels_data[str_idx]
            original_label = sample_labels[idx]
            
            # If reassessed labels is a list
            if isinstance(reassessed_labels, list):
                if len(reassessed_labels) == 0:
                    # No valid labels - uncertain/problematic
                    uncertain_samples.append(idx)
                elif original_label not in reassessed_labels:
                    # Original label not in reassessed list - mislabeled
                    sample_path = sample_paths[idx]
                    given_class = val_dataset.classes[original_label]
                    relative_path = os.path.relpath(sample_path, VAL_DIR)
                    
                    # Get reassessed class names
                    reassessed_classes = [val_dataset.classes[l] for l in reassessed_labels if l < len(val_dataset.classes)]
                    
                    mislabeled_samples.append({
                        'index': int(idx),
                        'file_path': sample_path,
                        'relative_path': relative_path,
                        'original_label': int(original_label),
                        'original_class': given_class,
                        'reassessed_labels': [int(l) for l in reassessed_labels],
                        'reassessed_classes': reassessed_classes,
                        'source': 'Google ReaL Labels',
                        'confidence': 'High (Multi-annotator consensus)'
                    })
                    
                    if len(reassessed_labels) > 1:
                        multi_label_samples.append(idx)
            else:
                # Single reassessed label
                if original_label != reassessed_labels:
                    sample_path = sample_paths[idx]
                    given_class = val_dataset.classes[original_label]
                    reassessed_class = val_dataset.classes[reassessed_labels]
                    relative_path = os.path.relpath(sample_path, VAL_DIR)
                    
                    mislabeled_samples.append({
                        'index': int(idx),
                        'file_path': sample_path,
                        'relative_path': relative_path,
                        'original_label': int(original_label),
                        'original_class': given_class,
                        'reassessed_label': int(reassessed_labels),
                        'reassessed_class': reassessed_class,
                        'source': 'Google ReaL Labels',
                        'confidence': 'High (Multi-annotator consensus)'
                    })
else:
    print("Warning: ReaL labels data not available")
    print("This script requires the official Google ReaL labels dataset")
    print("Please ensure internet connection or download manually from:")
    print("https://github.com/google-research/reassessed-imagenet")

num_issues = len(mislabeled_samples)
print(f"\n✓ Analysis completed")
print(f"  Mislabeled samples: {num_issues} ({num_issues/num_samples*100:.2f}%)")
print(f"  Multi-label samples: {len(multi_label_samples)}")
print(f"  Uncertain samples: {len(uncertain_samples)}")
print()

# Prepare summary
summary = {
    'total_samples': int(num_samples),
    'num_classes': int(num_classes),
    'num_mislabeled': int(num_issues),
    'num_multi_label': len(multi_label_samples),
    'num_uncertain': len(uncertain_samples),
    'percentage_mislabeled': float(num_issues/num_samples*100) if num_samples > 0 else 0,
    'source': 'Google Research - ReaL Labels (Reassessed ImageNet)',
    'paper': 'Are we done with ImageNet?',
    'paper_url': 'https://arxiv.org/abs/1902.10811',
    'github_url': 'https://github.com/google-research/reassessed-imagenet',
    'method': 'Multi-annotator reassessment with expert verification',
    'validation_directory': VAL_DIR
}

# Save JSON report
report = {
    'summary': summary,
    'mislabeled_samples': mislabeled_samples
}

print(f"Saving report to: {OUTPUT_JSON}")
with open(OUTPUT_JSON, 'w') as f:
    json.dump(report, f, indent=2)
print(f"✓ JSON report saved")

# Save text summary
print(f"Saving summary to: {OUTPUT_TXT}")
with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("ImageNet Validation - Google ReaL Labels Reassessment\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("SOURCE\n")
    f.write("-" * 80 + "\n")
    f.write("Paper: Are we done with ImageNet?\n")
    f.write("Authors: Lucas Beyer, Olivier J. Hénaff, Alexander Kolesnikov, et al.\n")
    f.write("Organization: Google Research\n")
    f.write("Published: 2020\n")
    f.write("URL: https://arxiv.org/abs/1902.10811\n")
    f.write("GitHub: https://github.com/google-research/reassessed-imagenet\n")
    f.write("\n")
    
    f.write("SUMMARY STATISTICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total samples:           {num_samples:,}\n")
    f.write(f"Number of classes:       {num_classes}\n")
    f.write(f"Mislabeled samples:      {num_issues:,} ({num_issues/num_samples*100:.2f}%)\n")
    f.write(f"Multi-label samples:     {len(multi_label_samples):,}\n")
    f.write(f"Uncertain samples:       {len(uncertain_samples):,}\n")
    f.write(f"Source:                  Google ReaL Labels (Multi-annotator)\n")
    f.write(f"Confidence:              High (Expert verification)\n")
    f.write("\n")
    
    f.write("METHODOLOGY\n")
    f.write("-" * 80 + "\n")
    f.write("- Complete re-annotation of ImageNet validation set\n")
    f.write("- Multiple expert annotators per image\n")
    f.write("- Consensus-based label assignment\n")
    f.write("- Identifies both mislabeled and multi-label images\n")
    f.write("- This list is INDEPENDENT of any model's predictions\n")
    f.write("\n")
    
    if num_issues > 0:
        f.write("MISLABELED SAMPLES (First 100)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'Original Class':<20} {'Reassessed Class':<20} {'File'}\n")
        f.write("-" * 80 + "\n")
        
        for sample in mislabeled_samples[:100]:
            reassessed = ""
            if 'reassessed_class' in sample:
                reassessed = sample['reassessed_class']
            elif 'reassessed_classes' in sample and len(sample['reassessed_classes']) > 0:
                reassessed = sample['reassessed_classes'][0]
            
            f.write(f"{sample['index']:<8} "
                    f"{sample['original_class']:<20} "
                    f"{reassessed:<20} "
                    f"{sample['relative_path']}\n")
        
        if num_issues > 100:
            f.write(f"\n... and {num_issues - 100} more (see JSON file for complete list)\n")
    
    f.write("\n")
    f.write("NOTES\n")
    f.write("-" * 80 + "\n")
    f.write("- ReaL labels provide more accurate ground truth than original labels\n")
    f.write("- Some images legitimately belong to multiple classes\n")
    f.write("- Removing mislabeled samples improves benchmark reliability\n")
    f.write("- Using ReaL labels for evaluation gives more accurate model performance\n")

print(f"✓ Summary saved")
print()

# Print summary
print("=" * 80)
print("DETECTION COMPLETED")
print("=" * 80)
print(f"Total samples analyzed:      {num_samples:,}")
print(f"Mislabeled samples found:    {num_issues:,} ({num_issues/num_samples*100:.2f}%)")
print(f"Multi-label samples:         {len(multi_label_samples):,}")
print(f"Uncertain samples:           {len(uncertain_samples):,}")
print()
print(f"Source: Google ReaL Labels (Multi-annotator Consensus)")
print(f"Paper: https://arxiv.org/abs/1902.10811")
print()
print(f"Results saved to:")
print(f"  - {OUTPUT_JSON}")
print(f"  - {OUTPUT_TXT}")
print()
print("Use 'move_mislabeled_samples.py' to move these samples")
print("(Update the REPORT_JSON variable to use this file)")
print("=" * 80)

