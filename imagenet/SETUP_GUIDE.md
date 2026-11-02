# ImageNet Dataset Setup Guide

This guide will help you organize your ImageNet dataset to match the code structure with minimal changes.

## Current Structure

You have:
- `ILSVRC2012_img_train/` - 1000 tar files (one per class)
- `ILSVRC2012_img_val/` - 50000 JPEG images (flat structure)

## Target Structure (What the Code Expects)

```
./
├── train/
│   ├── n01440764/
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   ├── n01443537/
│   │   └── ...
│   └── ... (1000 classes)
│
└── val/
    ├── n01440764/
    │   ├── ILSVRC2012_val_00000293.JPEG
    │   └── ...
    ├── n01443537/
    │   └── ...
    └── ... (1000 classes)
```

## Quick Setup (Recommended)

### Option 1: Automatic Setup Script

Simply run:
```bash
python setup_imagenet.py
```

This script will:
1. Extract all training tar files to `train/` directory
2. Ask about validation organization (if you have ground truth file)

### Option 2: Manual Setup

#### Step 1: Extract Training Data

```bash
python prepare_imagenet_data.py --extract-train --train-tar-dir ILSVRC2012_img_train --output-train-dir train
```

This will extract all tar files and create the `train/` directory with class subdirectories.

#### Step 2: Organize Validation Data

You need the validation ground truth file. If you have it:

```bash
python prepare_imagenet_data.py --organize-val --val-dir ILSVRC2012_img_val --output-val-dir val --ground-truth ILSVRC2012_validation_ground_truth.txt
```

**Getting the Ground Truth File:**

1. Download `ILSVRC2012_devkit_t12.tar.gz` from the ImageNet website
2. Extract it
3. Find `ILSVRC2012_validation_ground_truth.txt` in the devkit
4. Place it in your project directory

**Alternative: Using Mapping File**

If you have a mapping file (format: `image_name class_name`):

```bash
python prepare_imagenet_data.py --organize-val --val-dir ILSVRC2012_img_val --output-val-dir val --val-mapping val_mapping.txt
```

## After Setup

Once the data is organized, you can run training with:

```bash
python main_when.py --data . --optimizer adabelief --lr 1e-3
```

The `--data .` flag tells the code to use the current directory, where `train/` and `val/` folders are located.

## Verification

After setup, verify the structure:

```bash
# Check training data
ls train/ | wc -l  # Should show 1000 classes

# Check validation data
ls val/ | wc -l  # Should show 1000 classes

# Check a sample class
ls train/n01440764/ | head -5  # Should show images
```

## Troubleshooting

### Training data extraction is slow
- This is normal. Extracting 1000 tar files takes time and requires ~138 GB disk space.

### Validation organization fails
- Make sure you have the ground truth file
- Verify the ground truth file has 50000 lines (one label per validation image)
- Labels should be 1-indexed (1 to 1000)

### Not enough disk space
- Training data: ~138 GB
- Validation data: ~6.3 GB
- Total: ~144 GB

### Code still can't find data
- Make sure `train/` and `val/` folders are in the same directory as `main_when.py`
- Use `--data .` to specify current directory
- Or use `--data /path/to/parent/directory` if folders are elsewhere

## Notes

- The code expects class names to match between train and val directories
- Class names are derived from tar file names (e.g., `n01440764.tar` → `n01440764/`)
- Validation images keep their original names (e.g., `ILSVRC2012_val_00000001.JPEG`)

