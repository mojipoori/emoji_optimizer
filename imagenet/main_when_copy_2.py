from __future__ import print_function

import argparse
import json
import os
import random
import shutil
import time
import warnings
from functools import partial

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torchvision.datasets.folder import default_loader
from scipy.io import loadmat

from adabound import AdaBound
from shufflenet import *
from senet import *
from SGD_GC import SGD_GC
from adabelief_pytorch import AdaBelief
from AdamW import AdamW
from RAdam import RAdam
from emoji import Emoji


# =============================================================================
# Model names
# =============================================================================

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(models.__dict__[name])
)


# =============================================================================
# Arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="PyTorch ImageNet Training and Original/ReaL Evaluation"
)

parser.add_argument(
    "--data",
    metavar="DIR",
    default=".",
    help="path to dataset and evaluation metadata",
)

# ReaL / flat validation arguments.
parser.add_argument(
    "--val-folder",
    default="ILSVRC2012_img_val",
    type=str,
    help="Original flat ImageNet validation folder (50,000 JPEG files)",
)
parser.add_argument(
    "--real-labels",
    default="Google_ReaL_labels.json",
    type=str,
    help="Google ReaL labels JSON file",
)
parser.add_argument(
    "--imagenet-val-gt",
    default="ILSVRC2012_validation_ground_truth.txt",
    type=str,
    help="Original ImageNet 2012 validation ground-truth file",
)
parser.add_argument(
    "--imagenet-meta",
    default=os.path.join("ILSVRC2012_devkit_t12", "data", "meta.mat"),
    type=str,
    help="ImageNet devkit meta.mat file",
)

parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet18",
    help="model architecture: "
         + " | ".join(model_names)
         + " (default: resnet18)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=1,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 1)",
)
parser.add_argument(
    "--optimizer",
    default="adabelief",
    type=str,
    help="optimizer to use (sgd, adabound, adabelief, adamw, emoji)",
)
parser.add_argument(
    "--centralize",
    default=False,
    dest="centralize",
    action="store_true",
    help="use gradient centralization (SGD_GC)",
)
parser.add_argument(
    "--reset",
    default=False,
    dest="reset",
    action="store_true",
    help="reset optimizer state when learning rate is decayed",
)
parser.add_argument(
    "--warmup",
    default=False,
    dest="warmup",
    action="store_true",
    help="use linear learning-rate warmup",
)
parser.add_argument(
    "--reset_resume_optim",
    default=False,
    dest="reset_resume_optim",
    action="store_true",
    help="do not load optimizer state when resuming",
)
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--when",
    nargs="+",
    type=int,
    default=[-1],
    help="epochs at which to decay the learning rate by 10x",
)
parser.add_argument(
    "--decay_epoch",
    default=30,
    type=int,
    metavar="N",
    help="legacy decay interval used by checkpoint saving",
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.001,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    metavar="M",
    help="momentum (SGD)",
)
parser.add_argument(
    "--fixed_decay",
    action="store_true",
    help="use fixed weight decay (AdaBelief)",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-2,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-2)",
    dest="weight_decay",
)
parser.add_argument(
    "--l1_decay",
    default=0.0,
    type=float,
    help="L1 weight decay (currently unused)",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to checkpoint",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate checkpoint using original ImageNet, Google ReaL, and hybrid labels",
)
parser.add_argument(
    "--pretrained",
    dest="pretrained",
    action="store_true",
    help="use pre-trained model",
)
parser.add_argument(
    "--world-size",
    default=-1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument(
    "--rank",
    default=-1,
    type=int,
    help="node rank for distributed training",
)
parser.add_argument(
    "--dist-url",
    default="tcp://224.66.41.62:23456",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend",
    default="nccl",
    type=str,
    help="distributed backend",
)
parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="seed for initializing training",
)
parser.add_argument(
    "--gpu",
    default=None,
    type=int,
    help="GPU id to use",
)
parser.add_argument(
    "--multiprocessing-distributed",
    action="store_true",
    help="use multi-processing distributed training",
)

# AdaBelief / related optimizer arguments.
parser.add_argument("--eps", default=1e-8, type=float, help="epsilon")
parser.add_argument("--beta1", default=0.9, type=float, help="beta1")
parser.add_argument("--beta2", default=0.999, type=float, help="beta2")
parser.add_argument(
    "--weight_decouple",
    default=True,
    type=bool,
    help="weight decouple (AdaBelief)",
)

# Emoji arguments: keep this optimizer API unchanged.
parser.add_argument(
    "--emoji_betas",
    type=float,
    nargs=2,
    default=[0.9, 0.999],
    help="emoji betas (beta1, beta2)",
)
parser.add_argument(
    "--emoji_eps",
    type=float,
    default=1e-8,
    help="emoji epsilon",
)
parser.add_argument(
    "--emoji_weight_decouple",
    type=bool,
    default=False,
    help="use decoupled weight decay for emoji",
)
parser.add_argument(
    "--emoji_amsgrad",
    type=bool,
    default=False,
    help="use AMSGrad variant for emoji",
)


best_acc1 = 0


# =============================================================================
# Flat ImageNet validation dataset
# =============================================================================

class FlatImageNetValidationDataset(torch.utils.data.Dataset):
    """Load the original flat ImageNet validation folder."""

    IMG_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.loader = default_loader

        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                "Validation directory not found: {}".format(self.root)
            )

        filenames = sorted(
            filename
            for filename in os.listdir(self.root)
            if os.path.isfile(os.path.join(self.root, filename))
            and os.path.splitext(filename)[1] in self.IMG_EXTENSIONS
        )

        if len(filenames) != 50000:
            raise ValueError(
                "Expected exactly 50,000 validation images in '{}', "
                "but found {:,}.".format(self.root, len(filenames))
            )

        expected_filenames = [
            "ILSVRC2012_val_{:08d}.JPEG".format(i)
            for i in range(1, 50001)
        ]

        if [name.lower() for name in filenames] != [
            name.lower() for name in expected_filenames
        ]:
            raise ValueError(
                "Validation files are not the complete expected sequence "
                "from ILSVRC2012_val_00000001.JPEG to "
                "ILSVRC2012_val_00050000.JPEG."
            )

        self.samples = [
            (os.path.join(self.root, filename), -1)
            for filename in filenames
        ]
        self.filenames = filenames

        print(
            "[OK] Flat validation dataset loaded: {:,} images".format(
                len(self.samples)
            )
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, -1


# =============================================================================
# Resolve metadata file paths
# =============================================================================

def resolve_data_file(data_root, requested_path, fallback_paths=None):
    """Resolve a metadata path from --data or an absolute path."""

    if os.path.isabs(requested_path):
        if os.path.isfile(requested_path):
            return requested_path
        raise FileNotFoundError("File not found: {}".format(requested_path))

    candidates = [os.path.join(data_root, requested_path)]

    if fallback_paths:
        for fallback_path in fallback_paths:
            if os.path.isabs(fallback_path):
                candidates.append(fallback_path)
            else:
                candidates.append(os.path.join(data_root, fallback_path))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find requested file. Tried:\n" + "\n".join(candidates)
    )


# =============================================================================
# Load original ImageNet ground truth
# =============================================================================

def load_original_imagenet_targets(gt_file, meta_file):
    """
    Convert official ImageNet validation synset IDs to zero-based class IDs.

    The mapping follows torchvision ImageFolder class ordering:
    sorted WNIDs -> class indices 0,...,999.
    """

    print("\nLoading original ImageNet validation labels...")

    if not os.path.isfile(gt_file):
        raise FileNotFoundError(
            "Original ImageNet validation ground-truth file not found: {}".format(
                gt_file
            )
        )

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_ids = [int(line.strip()) for line in f if line.strip()]

    if len(gt_ids) != 50000:
        raise ValueError(
            "Expected 50,000 original ImageNet ground-truth entries, "
            "but found {:,}.".format(len(gt_ids))
        )

    if not os.path.isfile(meta_file):
        raise FileNotFoundError(
            "ImageNet devkit meta.mat not found: {}".format(meta_file)
        )

    meta = loadmat(meta_file, squeeze_me=True)["synsets"]

    # ImageNet leaf classes have zero children.
    leaf_meta = [row for row in meta if int(row[4]) == 0]

    if len(leaf_meta) != 1000:
        raise ValueError(
            "Expected 1,000 ImageNet leaf classes in meta.mat, "
            "but found {:,}.".format(len(leaf_meta))
        )

    id_to_wnid = {
        int(row[0]): str(row[1])
        for row in leaf_meta
    }

    sorted_wnids = sorted(id_to_wnid.values())
    wnid_to_class_idx = {
        wnid: idx for idx, wnid in enumerate(sorted_wnids)
    }

    original_targets = []

    for position, synset_id in enumerate(gt_ids):
        if synset_id not in id_to_wnid:
            raise ValueError(
                "Ground-truth synset ID {} at validation position {} "
                "was not found in meta.mat.".format(
                    synset_id, position + 1
                )
            )

        wnid = id_to_wnid[synset_id]

        if wnid not in wnid_to_class_idx:
            raise ValueError(
                "WNID {} at validation position {} was not found "
                "in the 1,000 ImageNet classes.".format(
                    wnid, position + 1
                )
            )

        original_targets.append(wnid_to_class_idx[wnid])

    print(
        "[OK] Loaded {:,} original ImageNet validation labels".format(
            len(original_targets)
        )
    )

    return original_targets


# =============================================================================
# Original ImageNet + Google ReaL + Hybrid evaluation
# =============================================================================

def validate_real_hybrid(val_loader, model, args):
    """
    Evaluate the same 50,000 images in three ways:

    1. Original ImageNet: original GT on all 50,000 images.
    2. ReaL-only: non-empty ReaL labels only.
    3. Hybrid: ReaL labels when present; original GT for ReaL == [].

    Hybrid accuracy is a custom metric, not official ReaL accuracy.
    """

    print("\n" + "=" * 80)
    print("ImageNet Original + Google ReaL + Hybrid Evaluation")
    print("=" * 80)

    real_labels_path = resolve_data_file(
        args.data,
        args.real_labels,
    )

    gt_path = resolve_data_file(
        args.data,
        args.imagenet_val_gt,
        fallback_paths=[
            "ILSVRC2012_validation_ground_truth.txt",
            os.path.join(
                "ILSVRC2012_devkit_t12",
                "data",
                "ILSVRC2012_validation_ground_truth.txt",
            ),
        ],
    )

    meta_path = resolve_data_file(
        args.data,
        args.imagenet_meta,
        fallback_paths=[
            os.path.join("ILSVRC2012_devkit_t12", "data", "meta.mat"),
            "meta.mat",
        ],
    )

    print("=> ReaL labels:       {}".format(real_labels_path))
    print("=> Original GT:       {}".format(gt_path))
    print("=> ImageNet meta.mat: {}".format(meta_path))

    # -------------------------------------------------------------------------
    # ReaL labels
    # -------------------------------------------------------------------------

    with open(real_labels_path, "r", encoding="utf-8") as f:
        real_labels = json.load(f)

    if not isinstance(real_labels, list):
        raise ValueError(
            "Google ReaL labels must contain a list of label lists."
        )

    if len(real_labels) != 50000:
        raise ValueError(
            "Expected 50,000 ReaL entries, but found {:,}.".format(
                len(real_labels)
            )
        )

    empty_label_count = 0
    multi_label_count = 0

    for i, labels in enumerate(real_labels):
        if not isinstance(labels, list):
            raise ValueError(
                "Invalid ReaL entry at index {}: expected a list.".format(i)
            )

        if len(labels) == 0:
            empty_label_count += 1

        if len(labels) > 1:
            multi_label_count += 1

        for label in labels:
            if not isinstance(label, int) or isinstance(label, bool):
                raise ValueError(
                    "Invalid ReaL label at entry {}: {!r} is not an integer.".format(
                        i, label
                    )
                )

            if label < 0 or label >= 1000:
                raise ValueError(
                    "Invalid ReaL class index {} at entry {}.".format(
                        label, i
                    )
                )

    print("[OK] ReaL JSON structure validated")
    print("[OK] Empty-label samples: {:,}".format(empty_label_count))
    print("[OK] Multi-label samples: {:,}".format(multi_label_count))

    # -------------------------------------------------------------------------
    # Original ImageNet targets
    # -------------------------------------------------------------------------

    original_targets = load_original_imagenet_targets(
        gt_file=gt_path,
        meta_file=meta_path,
    )

    dataset = val_loader.dataset
    num_samples = len(dataset)

    if num_samples != 50000:
        raise ValueError(
            "Expected exactly 50,000 validation images, but found {:,}.".format(
                num_samples
            )
        )

    if not hasattr(dataset, "samples"):
        raise ValueError(
            "Validation dataset does not provide .samples."
        )

    if len(dataset.samples) != num_samples:
        raise ValueError(
            "Dataset samples and dataset length do not match."
        )

    print(
        "[OK] Validation dataset contains {:,} images".format(num_samples)
    )

    # Counters: Original ImageNet, all 50,000.
    original_full_correct1 = 0
    original_full_correct5 = 0

    # Counters: ReaL-only, non-empty entries.
    real_evaluated = 0
    real_correct1 = 0
    real_correct5 = 0

    # Counters: empty-label fallback subset.
    fallback_count = 0
    fallback_correct1 = 0
    fallback_correct5 = 0

    # Counters: Hybrid, all 50,000.
    hybrid_correct1 = 0
    hybrid_correct5 = 0

    processed = 0
    sample_idx = 0
    evaluation_start_time = time.time()

    model.eval()

    with torch.no_grad():
        for batch_idx, (images, _dummy_target) in enumerate(val_loader):
            batch_start_time = time.time()

            output = model(images)

            if output.ndim != 2 or output.shape[1] != 1000:
                raise ValueError(
                    "Expected model output with shape [batch_size, 1000], "
                    "but received {}.".format(tuple(output.shape))
                )

            _, pred = output.topk(
                5,
                dim=1,
                largest=True,
                sorted=True,
            )
            pred = pred.cpu()

            batch_size = images.size(0)

            for batch_index in range(batch_size):
                if sample_idx >= num_samples:
                    raise RuntimeError(
                        "Processed more images than exist in the dataset."
                    )

                img_path = dataset.samples[sample_idx][0]
                filename = os.path.basename(img_path)

                try:
                    file_num = int(
                        filename.split("_")[-1].split(".")[0]
                    )
                except (ValueError, IndexError):
                    raise ValueError(
                        "Could not extract validation number from filename: {}".format(
                            filename
                        )
                    )

                if not 1 <= file_num <= 50000:
                    raise ValueError(
                        "Invalid validation file number {} in {}.".format(
                            file_num, filename
                        )
                    )

                # This verifies that the loader's sorted order matches the
                # filename index expected by both label files.
                if file_num != sample_idx + 1:
                    raise ValueError(
                        "Validation order mismatch: dataset position {} "
                        "contains {}.".format(sample_idx + 1, filename)
                    )

                predictions = pred[batch_index].tolist()

                # Original ImageNet ground truth.
                original_label = original_targets[file_num - 1]

                if predictions[0] == original_label:
                    original_full_correct1 += 1

                if original_label in predictions[:5]:
                    original_full_correct5 += 1

                # Corresponding ReaL labels.
                reassessed_labels = real_labels[file_num - 1]
                processed += 1

                if reassessed_labels:
                    # ReaL-only: any accepted label counts as correct.
                    real_evaluated += 1

                    is_real_correct1 = predictions[0] in reassessed_labels
                    is_real_correct5 = any(
                        prediction in reassessed_labels
                        for prediction in predictions[:5]
                    )

                    if is_real_correct1:
                        real_correct1 += 1
                        hybrid_correct1 += 1

                    if is_real_correct5:
                        real_correct5 += 1
                        hybrid_correct5 += 1

                else:
                    # Hybrid fallback: use the original ImageNet label.
                    fallback_count += 1

                    is_fallback_correct1 = predictions[0] == original_label
                    is_fallback_correct5 = original_label in predictions[:5]

                    if is_fallback_correct1:
                        fallback_correct1 += 1
                        hybrid_correct1 += 1

                    if is_fallback_correct5:
                        fallback_correct5 += 1
                        hybrid_correct5 += 1

                sample_idx += 1

            # Progress output.
            batch_elapsed = time.time() - batch_start_time
            total_elapsed = time.time() - evaluation_start_time
            images_per_second = (
                batch_size / batch_elapsed if batch_elapsed > 0 else 0.0
            )
            progress_percent = 100.0 * sample_idx / num_samples

            print(
                "Evaluation: "
                "batch {:>3}/{} | "
                "processed {:>6,}/50,000 ({:6.2f}%) | "
                "Original {:>6,} top1 | "
                "ReaL {:>6,} | "
                "fallback {:>5,} | "
                "batch {:5.2f}s | "
                "{:7.1f} img/s | "
                "elapsed {:6.1f} min".format(
                    batch_idx + 1,
                    len(val_loader),
                    sample_idx,
                    progress_percent,
                    original_full_correct1,
                    real_evaluated,
                    fallback_count,
                    batch_elapsed,
                    images_per_second,
                    total_elapsed / 60.0,
                ),
                flush=True,
            )

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------

    if processed != 50000:
        raise RuntimeError(
            "Processed {:,} instead of 50,000.".format(processed)
        )

    if real_evaluated != 50000 - empty_label_count:
        raise RuntimeError(
            "ReaL evaluated count does not match the non-empty ReaL entries."
        )

    if fallback_count != empty_label_count:
        raise RuntimeError(
            "Fallback count does not match the empty ReaL entries."
        )

    if real_evaluated + fallback_count != 50000:
        raise RuntimeError(
            "ReaL + fallback counts do not equal 50,000."
        )

    # -------------------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------------------

    original_full_acc1 = 100.0 * original_full_correct1 / 50000
    original_full_acc5 = 100.0 * original_full_correct5 / 50000

    real_acc1 = (
        100.0 * real_correct1 / real_evaluated
        if real_evaluated > 0 else float("nan")
    )
    real_acc5 = (
        100.0 * real_correct5 / real_evaluated
        if real_evaluated > 0 else float("nan")
    )

    fallback_acc1 = (
        100.0 * fallback_correct1 / fallback_count
        if fallback_count > 0 else float("nan")
    )
    fallback_acc5 = (
        100.0 * fallback_correct5 / fallback_count
        if fallback_count > 0 else float("nan")
    )

    hybrid_acc1 = 100.0 * hybrid_correct1 / 50000
    hybrid_acc5 = 100.0 * hybrid_correct5 / 50000

    total_elapsed = time.time() - evaluation_start_time

    # -------------------------------------------------------------------------
    # Print final results
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL IMAGENET + REAL + HYBRID EVALUATION RESULTS")
    print("=" * 80)

    print("\nORIGINAL IMAGENET GROUND TRUTH - ALL 50,000")
    print("-" * 80)
    print("Total validation images:       {:,}".format(50000))
    print("Original GT correct Top-1:     {:,}".format(original_full_correct1))
    print("Original GT correct Top-5:     {:,}".format(original_full_correct5))
    print("Original ImageNet Acc@1:       {:.3f}".format(original_full_acc1))
    print("Original ImageNet Acc@5:       {:.3f}".format(original_full_acc5))

    print("\nReaL-ONLY EVALUATION")
    print("-" * 80)
    print("ReaL-evaluated images:          {:,}".format(real_evaluated))
    print("ReaL correct Top-1:             {:,}".format(real_correct1))
    print("ReaL correct Top-5:             {:,}".format(real_correct5))
    print("ReaL Acc@1:                     {:.3f}".format(real_acc1))
    print("ReaL Acc@5:                     {:.3f}".format(real_acc5))

    print("\n[] FALLBACK EVALUATION")
    print("-" * 80)
    print("[] fallback images:             {:,}".format(fallback_count))
    print("Fallback correct Top-1:         {:,}".format(fallback_correct1))
    print("Fallback correct Top-5:         {:,}".format(fallback_correct5))
    print("Fallback Acc@1:                 {:.3f}".format(fallback_acc1))
    print("Fallback Acc@5:                 {:.3f}".format(fallback_acc5))

    print("\nHYBRID EVALUATION - ALL 50,000")
    print("-" * 80)
    print("Hybrid correct Top-1:           {:,}".format(hybrid_correct1))
    print("Hybrid correct Top-5:           {:,}".format(hybrid_correct5))
    print("Hybrid Acc@1:                   {:.3f}".format(hybrid_acc1))
    print("Hybrid Acc@5:                   {:.3f}".format(hybrid_acc5))

    print("\n" + "=" * 80)
    print(
        "Original ImageNet metrics use original ImageNet labels "
        "on all 50,000 images."
    )
    print(
        "ReaL metrics use non-empty ReaL labels and exclude "
        "images with empty ReaL labels."
    )
    print(
        "Hybrid metrics use ReaL where available and original "
        "ImageNet labels where ReaL == []."
    )
    print("Hybrid metrics are NOT official ReaL accuracy.")
    print("Total evaluation time:          {:.2f} min".format(total_elapsed / 60.0))
    print(
        "Check: ReaL + fallback =       {:,}".format(
            real_evaluated + fallback_count
        )
    )
    print("=" * 80)

    return (
        original_full_acc1,
        original_full_acc5,
        real_acc1,
        real_acc5,
        hybrid_acc1,
        hybrid_acc5,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        try:
            import numpy as np
            np.random.seed(args.seed)
        except ImportError:
            pass

        warnings.warn(
            "You have chosen to seed training. "
            "Weight initialization will be reproducible, but training "
            "operations use optimized algorithms for speed. "
            "Results may vary slightly between runs."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. "
            "This will completely disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = (
        args.world_size > 1 or args.multiprocessing_distributed
    )

    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(
            main_worker,
            nprocs=ngpus_per_node,
            args=(ngpus_per_node, args),
        )
    else:
        main_worker(args.gpu, ngpus_per_node, args)


# =============================================================================
# Data Prefetcher
# =============================================================================

class DataPrefetcher:
    def __init__(self, loader, stop_after=None):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()

        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input_tensor = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input_tensor, target

            if type(self.stop_after) is int and count > self.stop_after:
                break


def _worker_init_fn(worker_id, seed):
    """Initialize a data-loader worker with a reproducible seed."""
    worker_seed = seed % (2 ** 32) + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    try:
        import numpy as np
        np.random.seed(worker_seed)
    except ImportError:
        pass


# =============================================================================
# Main worker
# =============================================================================

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        try:
            import numpy as np
            np.random.seed(args.seed)
        except ImportError:
            pass

    # Keep the existing checkpoint naming convention unchanged.
    if args.optimizer == "emoji":
        filename = (
            "model-{}-optimizer-{}-lr-{}-epochs-{}-decay-epoch-{}-"
            "emoji_eps{}-emoji_beta1{}-emoji_beta2{}-start-epoch-{}-"
            "l2-decay{}-l1-decay{}-batch-{}-warmup-{}-fixed-decay-{}"
        ).format(
            args.arch,
            args.optimizer,
            args.lr,
            args.epochs,
            args.when,
            args.emoji_eps,
            args.emoji_betas[0],
            args.emoji_betas[1],
            args.start_epoch,
            args.weight_decay,
            args.l1_decay,
            args.batch_size,
            args.warmup,
            args.fixed_decay,
        )
    else:
        filename = (
            "model-{}-optimizer-{}-lr-{}-epochs-{}-decay-epoch-{}-"
            "eps{}-beta1{}-beta2{}-centralize-{}-reset{}-start-epoch-{}-"
            "l2-decay{}-l1-decay{}-batch-{}-warmup-{}-fixed-decay-{}"
        ).format(
            args.arch,
            args.optimizer,
            args.lr,
            args.epochs,
            args.when,
            args.eps,
            args.beta1,
            args.beta2,
            args.centralize,
            args.reset,
            args.start_epoch,
            args.weight_decay,
            args.l1_decay,
            args.batch_size,
            args.warmup,
            args.fixed_decay,
        )

    print(filename)
    csv_filename = filename + ".csv"

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # -------------------------------------------------------------------------
    # Distributed initialization
    # -------------------------------------------------------------------------

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # -------------------------------------------------------------------------
    # Create model
    # -------------------------------------------------------------------------

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))

        if args.arch == "shufflenet_v2_x0_5":
            model = shufflenet_v2_x0_5(pretrained=False)
        elif args.arch == "se_resnet18":
            model = se_resnet18()
        else:
            model = models.__dict__[args.arch]()

    # -------------------------------------------------------------------------
    # Move model to GPU / DataParallel
    # -------------------------------------------------------------------------

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(
                (args.workers + ngpus_per_node - 1) / ngpus_per_node
            )
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.gpu],
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # -------------------------------------------------------------------------
    # Optimizer selection: preserve the original Emoji / AdaBelief setup.
    # -------------------------------------------------------------------------

    if args.optimizer == "sgd" and not args.centralize:
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd" and args.centralize:
        optimizer = SGD_GC(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adabound":
        optimizer = AdaBound(
            model.parameters(),
            args.lr,
            eps=args.eps,
            betas=(args.beta1, args.beta2),
        )
    elif args.optimizer == "adabelief":
        optimizer = AdaBelief(
            model.parameters(),
            args.lr,
            eps=args.eps,
            betas=(args.beta1, args.beta2),
            weight_decouple=args.weight_decouple,
            weight_decay=args.weight_decay,
            fixed_decay=args.fixed_decay,
            rectify=False,
        )
    elif args.optimizer == "adamw":
        optimizer = AdamW(
            model.parameters(),
            args.lr,
            eps=args.eps,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "emoji":
        optimizer = Emoji(
            model.parameters(),
            lr=args.lr,
            betas=(args.emoji_betas[0], args.emoji_betas[1]),
            eps=args.emoji_eps,
            weight_decay=args.weight_decay,
            weight_decouple=args.emoji_weight_decouple,
            amsgrad=args.emoji_amsgrad,
        )
    else:
        raise ValueError("Optimizer not found: {}".format(args.optimizer))

    # -------------------------------------------------------------------------
    # Resume checkpoint
    # -------------------------------------------------------------------------

    val_acc1_history = []
    val_acc5_history = []

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if args.start_epoch is None:
                args.start_epoch = checkpoint["epoch"]

            csv_filename = (
                "{}-resume-from-epoch-{}.csv".format(
                    filename,
                    args.start_epoch,
                )
            )

            best_acc1 = checkpoint["best_acc1"]

            if args.gpu is not None and torch.is_tensor(best_acc1):
                best_acc1 = best_acc1.to(args.gpu)

            model.load_state_dict(checkpoint["state_dict"])

            if not args.reset_resume_optim:
                if "optimizer" not in checkpoint:
                    raise KeyError(
                        "Checkpoint does not contain optimizer state. "
                        "Use --reset_resume_optim to skip loading it."
                    )
                optimizer.load_state_dict(checkpoint["optimizer"])

            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume,
                    checkpoint["epoch"],
                )
            )
        else:
            raise FileNotFoundError(
                "No checkpoint found at '{}'".format(args.resume)
            )
    else:
        if args.start_epoch is None:
            args.start_epoch = 0

    cudnn.benchmark = True

    # =========================================================================
    # EVALUATION MODE: Original ImageNet + ReaL + Hybrid
    # =========================================================================

    if args.evaluate:
        if not args.resume:
            raise ValueError(
                "Evaluation requires --resume PATH to a trained checkpoint."
            )

        valdir = os.path.join(args.data, args.val_folder)

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        print("=> using flat validation folder: {}".format(valdir))

        val_dataset = FlatImageNetValidationDataset(
            valdir,
            transform=val_transform,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        val_loader = DataPrefetcher(val_loader)

        validate_real_hybrid(
            val_loader,
            model,
            args,
        )

        return

    # =========================================================================
    # TRAINING MODE: preserve the original ImageNet training pipeline
    # =========================================================================

    traindir = os.path.join(args.data, "ILSVRC2012_img_train")
    valdir = os.path.join(args.data, "ILSVRC2012_img_val_0")

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            seed=args.seed if args.seed is not None else 0,
        )
    else:
        train_sampler = None

    generator = None
    if args.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(args.seed)

    worker_init_fn = (
        partial(_worker_init_fn, seed=args.seed)
        if args.seed is not None
        else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    train_loader = DataPrefetcher(train_loader)
    val_loader = DataPrefetcher(val_loader)

    # If a CSV from an earlier run exists, preserve the original behavior of
    # starting a new history for this invocation. Evaluation does not write CSV.
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args,
        )

        val_acc1, val_acc5, _val_loss = validate(
            val_loader,
            model,
            criterion,
            args,
        )

        val_acc1_history.append(val_acc1.item())
        val_acc5_history.append(val_acc5.item())

        results = {
            "epoch": list(range(args.start_epoch, epoch + 1)),
            "val_acc1": val_acc1_history,
            "val_acc5": val_acc5_history,
        }

        pd.DataFrame(results).to_csv(
            csv_filename,
            index=False,
        )

        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if (
            not args.multiprocessing_distributed
            or (
                args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0
            )
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                filename=filename,
                epoch=epoch,
                decay_epoch=args.decay_epoch,
            )


# =============================================================================
# Train
# =============================================================================

def train(train_loader, model, criterion, optimizer, epoch, args):
    """Train for one epoch without computing training accuracy."""

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        losses.update(loss.detach().item(), images.size(0))

        if args.warmup and epoch % args.decay_epoch == 0:
            print("Warmup")
            lr = args.lr * (0.1 ** (epoch // args.decay_epoch))
            lr_tmp = lr / float(len(train_loader)) * float(i)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_tmp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg


# =============================================================================
# Conventional ImageNet validation (used during training)
# =============================================================================

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Validation: ",
    )

    model.eval()

    with torch.no_grad():
        end = time.time()

        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(
            " * Validation Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(
                top1=top1,
                top5=top5,
            )
        )

    return top1.avg, top5.avg, losses.avg


# =============================================================================
# Save checkpoint
# =============================================================================

def save_checkpoint(
    state,
    is_best,
    filename="checkpoint.pth.tar",
    epoch=0,
    decay_epoch=30,
):
    torch.save(state, filename)

    if is_best:
        shutil.copyfile(
            filename,
            filename + "_model_best.pth.tar",
        )

    if (epoch + 1) % decay_epoch == 0:
        torch.save(
            state,
            "{}-epoch-{}".format(filename, epoch),
        )


# =============================================================================
# Average Meter
# =============================================================================

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


# =============================================================================
# Progress Meter
# =============================================================================

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# =============================================================================
# Learning rate
# =============================================================================

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate at epochs listed in args.when."""

    if epoch in args.when:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= 0.1

        if args.reset:
            if not hasattr(optimizer, "reset"):
                raise AttributeError(
                    "--reset was requested, but this optimizer has no reset() method."
                )
            optimizer.reset()


# =============================================================================
# Accuracy
# =============================================================================

def accuracy(output, target, topk=(1,)):
    """Compute accuracy over the specified top-k predictions."""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = (
                correct[:k]
                .reshape(-1)
                .float()
                .sum(0, keepdim=True)
            )
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
