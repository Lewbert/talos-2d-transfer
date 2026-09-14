# TALOS one-dir build (no self-zipping — dist/TALOS is the deliverable).
# Usage:  pwsh -File packaging/build.ps1
conda activate talos
python -m PyInstaller --clean --noconfirm packaging/talos.spec
Write-Host "Build complete: dist/TALOS/talos.exe"
