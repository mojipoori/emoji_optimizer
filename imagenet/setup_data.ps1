# ImageNet Dataset Setup Script
# This script organizes your ImageNet data to match the code structure

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ImageNet Dataset Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python not found. Please install Python first." -ForegroundColor Red
    exit 1
}

# Run the setup script
Write-Host "Running setup script..." -ForegroundColor Yellow
Write-Host ""

python setup_imagenet.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Setup completed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run training with:" -ForegroundColor Yellow
    Write-Host "  python main_when.py --data . --optimizer adabelief --lr 1e-3" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Setup encountered errors. Please check the output above." -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
}

