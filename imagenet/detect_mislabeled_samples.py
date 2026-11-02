"""
Script to detect mislabeled samples in ImageNet validation dataset using cleanlab.
Uses trained ResNet18 model to generate predictions and cleanlab to identify label issues.
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
import time

# Configuration
CHECKPOINT_PATH = "model-resnet18-optimizer-adabelief-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-eps1e-08-beta10.9-beta20.999-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False_model_best.pth.tar"
VAL_DIR = "ILSVRC2012_img_val"
BATCH_SIZE = 256
NUM_WORKERS = 4
OUTPUT_JSON = "mislabeled_samples_report.json"
OUTPUT_TXT = "mislabeled_samples_summary.txt"

def main():
    print("=" * 80)
    print("ImageNet Validation Dataset - Mislabeled Sample Detection")
    print("=" * 80)
    print()

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("WARNING: CUDA not available, using CPU (this will be slower)")
    print()

    # Load model
    print("Loading ResNet18 model...")
    model = models.resnet18(pretrained=False, num_classes=1000)

    # Load checkpoint
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    # Handle DataParallel state dict
    state_dict = checkpoint['state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        # Model was saved with DataParallel, need to handle it
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            model.load_state_dict(state_dict)
            model = model.to(device)
        else:
            # Remove 'module.' prefix from keys
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.' prefix
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            model = model.to(device)
    else:
        model.load_state_dict(state_dict)
        model = model.to(device)

    model.eval()
    print(f"✓ Checkpoint loaded (best_acc1: {checkpoint.get('best_acc1', 'N/A')})")
    print()

    # Prepare data transforms (same as validation in main_when.py)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Load validation dataset
    print(f"Loading validation dataset from: {VAL_DIR}")
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    num_samples = len(val_dataset)
    num_classes = len(val_dataset.classes)
    print(f"✓ Dataset loaded: {num_samples} samples, {num_classes} classes")
    print()

    # Get all file paths and labels
    print("Collecting sample information...")
    sample_paths = [path for path, _ in val_dataset.samples]
    true_labels = np.array([label for _, label in val_dataset.samples])
    print(f"✓ Collected {len(sample_paths)} sample paths")
    print()

    # Run inference to get predicted probabilities
    print("Running inference on validation dataset...")
    print("This may take several minutes...")
    start_time = time.time()

    all_pred_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(val_loader, desc="Processing batches")):
            images = images.to(device, non_blocking=True)
            
            # Get model predictions
            outputs = model(images)
            
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            all_pred_probs.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all predictions
    pred_probs = np.vstack(all_pred_probs)
    targets = np.concatenate(all_targets)

    inference_time = time.time() - start_time
    print(f"\n✓ Inference completed in {inference_time:.2f} seconds ({inference_time/60:.2f} minutes)")
    print(f"  Prediction shape: {pred_probs.shape}")
    print()

    # Calculate basic accuracy
    predicted_labels = np.argmax(pred_probs, axis=1)
    accuracy = (predicted_labels == targets).mean() * 100
    print(f"Model accuracy on validation set: {accuracy:.2f}%")
    print()

    # Use cleanlab to find label issues
    print("Running cleanlab to detect mislabeled samples...")
    print("Using automatic threshold detection...")

    # Get label quality scores
    label_quality_scores = get_label_quality_scores(
        labels=targets,
        pred_probs=pred_probs,
        method='self_confidence'
    )

    # Find label issues (returns boolean mask)
    label_issues_mask = find_label_issues(
        labels=targets,
        pred_probs=pred_probs,
        filter_by='both'  # Use both low confidence and disagreement
    )

    num_issues = label_issues_mask.sum()
    print(f"\n✓ Cleanlab analysis completed")
    print(f"  Found {num_issues} potential label issues ({num_issues/num_samples*100:.2f}% of dataset)")
    print()

    # Get indices of mislabeled samples sorted by severity (lowest quality first)
    issue_indices = np.where(label_issues_mask)[0]
    issue_scores = label_quality_scores[issue_indices]
    sorted_issue_idx = issue_indices[np.argsort(issue_scores)]  # Sort by lowest quality first

    # Prepare detailed report
    print("Preparing detailed report...")
    mislabeled_samples = []

    for idx in sorted_issue_idx:
        sample_path = sample_paths[idx]
        given_label = targets[idx]
        predicted_label = predicted_labels[idx]
        quality_score = label_quality_scores[idx]
        given_class = val_dataset.classes[given_label]
        predicted_class = val_dataset.classes[predicted_label]
        
        # Get relative path from VAL_DIR
        relative_path = os.path.relpath(sample_path, VAL_DIR)
        
        mislabeled_samples.append({
            'index': int(idx),
            'file_path': sample_path,
            'relative_path': relative_path,
            'given_label': int(given_label),
            'given_class': given_class,
            'predicted_label': int(predicted_label),
            'predicted_class': predicted_class,
            'label_quality_score': float(quality_score),
            'prediction_confidence': float(pred_probs[idx, predicted_label]),
            'given_label_confidence': float(pred_probs[idx, given_label])
        })

    # Prepare summary statistics
    summary = {
        'total_samples': int(num_samples),
        'num_classes': int(num_classes),
        'num_mislabeled': int(num_issues),
        'percentage_mislabeled': float(num_issues/num_samples*100),
        'model_accuracy': float(accuracy),
        'inference_time_seconds': float(inference_time),
        'checkpoint_used': CHECKPOINT_PATH,
        'validation_directory': VAL_DIR,
        'detection_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save JSON report
    report = {
        'summary': summary,
        'mislabeled_samples': mislabeled_samples
    }

    print(f"Saving detailed report to: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ JSON report saved")

    # Save human-readable summary
    print(f"Saving summary to: {OUTPUT_TXT}")
    with open(OUTPUT_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ImageNet Validation Dataset - Mislabeled Sample Detection Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total samples:           {num_samples:,}\n")
        f.write(f"Number of classes:       {num_classes}\n")
        f.write(f"Model accuracy:          {accuracy:.2f}%\n")
        f.write(f"Mislabeled samples:      {num_issues:,} ({num_issues/num_samples*100:.2f}%)\n")
        f.write(f"Inference time:          {inference_time:.2f} seconds ({inference_time/60:.2f} minutes)\n")
        f.write(f"Detection timestamp:     {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint used:         {CHECKPOINT_PATH}\n")
        f.write("\n")
        
        f.write("TOP 50 MOST LIKELY MISLABELED SAMPLES\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Index':<8} {'Quality':<10} {'Given Class':<15} {'Predicted Class':<15} {'File'}\n")
        f.write("-" * 80 + "\n")
        
        for sample in mislabeled_samples[:50]:
            f.write(f"{sample['index']:<8} "
                    f"{sample['label_quality_score']:<10.4f} "
                    f"{sample['given_class']:<15} "
                    f"{sample['predicted_class']:<15} "
                    f"{sample['relative_path']}\n")
        
        if num_issues > 50:
            f.write(f"\n... and {num_issues - 50} more (see JSON file for complete list)\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("LABEL QUALITY SCORE DISTRIBUTION\n")
        f.write("=" * 80 + "\n")
        f.write("Lower scores indicate higher likelihood of mislabeling\n\n")
        
        # Score distribution
        all_scores = label_quality_scores
        f.write(f"Minimum score:     {all_scores.min():.4f}\n")
        f.write(f"25th percentile:   {np.percentile(all_scores, 25):.4f}\n")
        f.write(f"Median score:      {np.median(all_scores):.4f}\n")
        f.write(f"75th percentile:   {np.percentile(all_scores, 75):.4f}\n")
        f.write(f"Maximum score:     {all_scores.max():.4f}\n")
        f.write(f"Mean score:        {all_scores.mean():.4f}\n")
        f.write(f"Std deviation:     {all_scores.std():.4f}\n")
        f.write("\n")
        
        # Class distribution of mislabeled samples
        f.write("=" * 80 + "\n")
        f.write("CLASSES WITH MOST MISLABELED SAMPLES (Top 20)\n")
        f.write("=" * 80 + "\n")
        
        from collections import Counter
        given_class_counts = Counter([s['given_class'] for s in mislabeled_samples])
        f.write(f"{'Class':<20} {'Count':<10}\n")
        f.write("-" * 80 + "\n")
        for class_name, count in given_class_counts.most_common(20):
            f.write(f"{class_name:<20} {count:<10}\n")

    print(f"✓ Summary saved")
    print()

    # Print summary to console
    print("=" * 80)
    print("DETECTION COMPLETED")
    print("=" * 80)
    print(f"Total samples analyzed:      {num_samples:,}")
    print(f"Mislabeled samples found:    {num_issues:,} ({num_issues/num_samples*100:.2f}%)")
    print(f"Model accuracy:              {accuracy:.2f}%")
    print()
    print(f"Results saved to:")
    print(f"  - {OUTPUT_JSON}")
    print(f"  - {OUTPUT_TXT}")
    print()
    print("Top 10 most likely mislabeled samples:")
    print("-" * 80)
    print(f"{'Quality':<10} {'Given Class':<20} {'Predicted Class':<20}")
    print("-" * 80)
    for sample in mislabeled_samples[:10]:
        print(f"{sample['label_quality_score']:<10.4f} "
              f"{sample['given_class']:<20} "
              f"{sample['predicted_class']:<20}")
    print()
    print("Run 'move_mislabeled_samples.py' to move these samples to ILSVRC2012_img_valWRONG")
    print("=" * 80)


if __name__ == '__main__':
    main()

