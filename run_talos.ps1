param(
    [switch]$Sim
)
conda activate talos
if ($Sim) {
    python -m talos --sim
} else {
    python -m talos
}
