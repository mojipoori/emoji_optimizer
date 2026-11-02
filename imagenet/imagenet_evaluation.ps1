# ============================================================
# ImageNet ReaL evaluation
# Saves terminal output separately for each optimizer
# ============================================================

$commonArgs = @(
    "main_when_copy_2.py",
    "--data", ".",
    "--evaluate",
    "--val-folder", "ILSVRC2012_img_val",
    "--real-labels", "Google_ReaL_labels.json",
    "--when", "30", "60", "90",
    "--epochs", "100",
    "--lr", "0.001",
    "--batch-size", "256"
)
# ============================================================
# emoji
# ============================================================

$emojiBase = "model-resnet18-optimizer-emoji-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-emoji_eps1e-08-emoji_beta10.9-emoji_beta20.999-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False"

$emojiLog = "emoji_reallabel_evaluation.txt"

Write-Host "============================================================"
Write-Host "Starting emoji evaluation"
Write-Host "============================================================"
# Last checkpoint (epoch 100)
& python @commonArgs `
    "--optimizer", "emoji" `
    "--resume", "$emojiBase" 2>&1 |
    Tee-Object -FilePath $emojiLog
# Epoch 30 checkpoint (epoch-29)
& python @commonArgs `
    "--optimizer", "emoji" `
    "--resume", "$emojiBase-epoch-29" 2>&1 |
    Tee-Object -FilePath $emojiLog -Append

# Epoch 60 checkpoint (epoch-59)
& python @commonArgs `
    "--optimizer", "emoji" `
    "--resume", "$emojiBase-epoch-59" 2>&1 |
    Tee-Object -FilePath $emojiLog -Append

# Epoch 90 checkpoint (epoch-89)
& python @commonArgs `
    "--optimizer", "emoji" `
    "--resume", "$emojiBase-epoch-89" 2>&1 |
    Tee-Object -FilePath $emojiLog -Append

# Best checkpoint
& python @commonArgs `
    "--optimizer", "emoji" `
    "--resume", "${emojiBase}_model_best.pth.tar" 2>&1 |
    Tee-Object -FilePath $emojiLog -Append



# ============================================================
# AdaBelief
# ============================================================

$adaBase = "model-resnet18-optimizer-adabelief-lr-0.001-epochs-100-decay-epoch-[30, 60, 90]-eps1e-08-beta10.9-beta20.999-centralize-False-resetFalse-start-epoch-None-l2-decay0.01-l1-decay0.0-batch-256-warmup-False-fixed-decay-False"

$adaLog = "adabelief_reallabel_evaluation.txt"

Write-Host "============================================================"
Write-Host "Starting AdaBelief evaluation"
Write-Host "============================================================"
# Last checkpoint (epoch 100)
& python @commonArgs `
    "--optimizer", "adabelief" `
    "--resume", "$adaBase" 2>&1 |
    Tee-Object -FilePath $adaLog 
# Epoch 30 checkpoint (epoch-29)
& python @commonArgs `
    "--optimizer", "adabelief" `
    "--resume", "$adaBase-epoch-29" 2>&1 |
    Tee-Object -FilePath $adaLog -Append

# Epoch 60 checkpoint (epoch-59)
& python @commonArgs `
    "--optimizer", "adabelief" `
    "--resume", "$adaBase-epoch-59" 2>&1 |
    Tee-Object -FilePath $adaLog -Append

# Epoch 90 checkpoint (epoch-89)
& python @commonArgs `
    "--optimizer", "adabelief" `
    "--resume", "$adaBase-epoch-89" 2>&1 |
    Tee-Object -FilePath $adaLog -Append

# Best checkpoint
& python @commonArgs `
    "--optimizer", "adabelief" `
    "--resume", "${adaBase}_model_best.pth.tar" 2>&1 |
    Tee-Object -FilePath $adaLog -Append







Write-Host ""
Write-Host "============================================================"
Write-Host "All evaluations completed."
Write-Host "AdaBelief log: $adaLog"
Write-Host "emoji log:     $emojiLog"
Write-Host "============================================================"