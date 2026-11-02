"""
Organize ImageNet ILSVRC2012 validation images into WNID subfolders using devkit.
Requirements:
- ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
- ILSVRC2012_devkit_t12/data/meta.mat
- ILSVRC2012_img_val/ with flat validation images
"""

import os
import sys
import shutil
from pathlib import Path


def ensure_scipy():
    try:
        import scipy.io  # noqa: F401
        return True
    except Exception:
        print("scipy not found. Installing scipy ...")
        try:
            # Non-interactive install
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "scipy"])
            import scipy.io  # noqa: F401
            return True
        except Exception as e:
            print(f"Failed to install scipy automatically: {e}")
            return False


def load_id_to_wnid_from_meta(meta_mat_path: Path):
    """
    Returns a dict mapping ILSVRC2012_ID (1..1000 for leaves) -> WNID (e.g., 'n01440764')
    """
    from scipy.io import loadmat

    mat = loadmat(str(meta_mat_path), squeeze_me=True, struct_as_record=False)
    # 'synsets' is an array of structs with fields: ILSVRC2012_ID, WNID, ...
    synsets = mat.get("synsets", None)
    if synsets is None:
        raise RuntimeError("Could not find 'synsets' in meta.mat")

    id_to_wnid = {}
    for entry in synsets:
        # entry.ILSVRC2012_ID is an int, entry.WNID is a string
        ilsvrc_id = int(entry.ILSVRC2012_ID)
        wnid = str(entry.WNID)
        id_to_wnid[ilsvrc_id] = wnid
    return id_to_wnid


def organize_val(val_dir: Path, devkit_dir: Path):
    gt_file = devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
    meta_mat = devkit_dir / "data" / "meta.mat"

    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_file}")
    if not meta_mat.exists():
        raise FileNotFoundError(f"meta.mat not found: {meta_mat}")

    if not ensure_scipy():
        raise RuntimeError("scipy is required to read meta.mat. Please install scipy and retry.")

    id_to_wnid = load_id_to_wnid_from_meta(meta_mat)

    # Read labels (1-indexed per devkit)
    with open(gt_file, "r") as f:
        labels = [int(line.strip()) for line in f if line.strip()]

    # Collect images sorted by filename as specified by devkit
    images = sorted(list(val_dir.glob("*.JPEG")) + list(val_dir.glob("*.jpg")))

    if len(images) != len(labels):
        print(f"Warning: images ({len(images)}) != labels ({len(labels)}). Proceeding with min length.")
        n = min(len(images), len(labels))
        images = images[:n]
        labels = labels[:n]

    # Move images into val/<WNID>/
    created_dirs = set()
    moved = 0
    for img_path, label in zip(images, labels):
        wnid = id_to_wnid.get(label, None)
        if wnid is None:
            print(f"Warning: No WNID for label {label}; skipping {img_path.name}")
            continue
        class_dir = val_dir / wnid
        if wnid not in created_dirs:
            class_dir.mkdir(exist_ok=True)
            created_dirs.add(wnid)
        dest = class_dir / img_path.name
        # Use move to avoid duplicating storage
        shutil.move(str(img_path), str(dest))
        moved += 1

    print(f"Organized {moved} images into {len(created_dirs)} class folders under: {val_dir}")
    return moved, len(created_dirs)


def main():
    base = Path(".")
    val_dir = base / "ILSVRC2012_img_val"
    devkit_dir = base / "ILSVRC2012_devkit_t12"

    if not val_dir.exists():
        print(f"Error: Validation directory not found: {val_dir}")
        return 1
    if not devkit_dir.exists():
        print(f"Error: Devkit directory not found: {devkit_dir}")
        return 1

    try:
        moved, num_classes = organize_val(val_dir, devkit_dir)
        print(f"Done. Moved: {moved}, Classes: {num_classes}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())


