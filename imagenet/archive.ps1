# ============================================================
# پاک‌سازی فایل‌های غیرضروری پروژه ImageNet
# ============================================================

# ایجاد پوشه آرشیو
New-Item -ItemType Directory -Force -Path "archive_unnecessary"

Write-Host "شروع انتقال فایل‌های غیرضروری..." -ForegroundColor Cyan

# --------------------------------------------------
# 1. نسخه‌های قدیمی اسکریپت اصلی
# --------------------------------------------------
$oldMains = @(
    "main_when.py",
    "main_when copy.py",
    "main_when_01.py"
)

foreach ($f in $oldMains) {
    if (Test-Path $f) {
        Move-Item $f "archive_unnecessary\" -Force
        Write-Host "منتقل شد: $f"
    }
}

# --------------------------------------------------
# 2. اسکریپت‌های دیباگ و موقت
# --------------------------------------------------
$debugScripts = @(
    "check_cv.py",
    "check_data_integrity.py",
    "check_duplicates.py",
    "check_validation_labels.py",
    "compare_val_dirs.py",
    "fix_validation.py",
    "fix_val_final.py",
    "delete_column_from_csv.py",
    "deepseek_python_20260717_23ea90.py",
    "graph.py",
    "visualize_readable.py",
    "visualize_validation.py",
    "read_checkpoint.py",
    "verify_label_mapping.py",
    "final_data_report.py",
    "export"
)

foreach ($f in $debugScripts) {
    if (Test-Path $f) {
        Move-Item $f "archive_unnecessary\" -Force
        Write-Host "منتقل شد: $f"
    }
}

# --------------------------------------------------
# 3. پوشه __pycache__
# --------------------------------------------------
if (Test-Path "__pycache__") {
    Move-Item "__pycache__" "archive_unnecessary\" -Force
    Write-Host "منتقل شد: __pycache__"
}

# --------------------------------------------------
# 4. پوشه‌های آزمایشی داخل runs (نام‌های غیر استاندارد)
# --------------------------------------------------
$runsToArchive = @(
    "runs\model-resnet18-optimizer-adabelief-lr-0.001-epochs-100-decay-epoch-(30_60_90)-eps1e-08-beta10.9-beta20.999-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False",
    "runs\model-resnet18-optimizer-adabelief-lr-0.001-epochs-100-decay-epoch-30_60_90-eps1e-08-beta10.9-beta20.999-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False",
    "runs\model-resnet18-optimizer-adabelief-lr-0.001-epochs-100-decay-epoch-[-1]-eps1e-08-beta10.9-beta20.999-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False",
    "runs\model-resnet18-optimizer-adabelief-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-eps1e-08-beta10.9-beta20.999-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-128-warmup-False-fixed-decay-False",
    "runs\model-resnet18-optimizer-adabelief-lr-0.001-epochs-90-decay-epoch-[30, 60, 90]-eps1e-08-beta10.9-beta20.999-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False",
    "runs\model-resnet18-optimizer-mojic-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-mojic_eps1e-08-mojic_beta10.9-mojic_beta20.9-mojic_beta30.999-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False"
)

foreach ($dir in $runsToArchive) {
    if (Test-Path $dir) {
        Move-Item $dir "archive_unnecessary\" -Force
        Write-Host "منتقل شد: $dir"
    }
}

# --------------------------------------------------
# 5. پوشه دانلود موقت (در صورت وجود)
# --------------------------------------------------
if (Test-Path "ILSVRC2012_val_download") {
    Move-Item "ILSVRC2012_val_download" "archive_unnecessary\" -Force
    Write-Host "منتقل شد: ILSVRC2012_val_download"
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "پاک‌سازی تمام شد."
Write-Host "فایل‌های غیرضروری به پوشه archive_unnecessary منتقل شدند."
Write-Host "============================================================" -ForegroundColor Green