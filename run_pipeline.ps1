param([switch]$ComparePreprocessing)
$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }
Set-Location $PSScriptRoot
function Run-Step([string]$Script) { Write-Host "`n=== $Script ===" -ForegroundColor Cyan; & $Python $Script; if ($LASTEXITCODE -ne 0) { throw "$Script başarısız." } }
Run-Step "src/01_data_audit.py"
Run-Step "src/02_dataset_preview.py"
Run-Step "src/04_prepare_splits.py"
if (-not (Test-Path "results/tables/split_manifest.csv")) { Write-Warning "Eğitime uygun etiketli veri yok; veri problem raporuna bakın."; exit 0 }
& $Python -c "import torch,sys; print('CUDA:',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'yok'); sys.exit(0 if torch.cuda.is_available() else 2)"
if ($LASTEXITCODE -ne 0) { throw "CUDA gerekli; eğitim güvenli biçimde durduruldu." }
Run-Step "src/07_train.py"; Run-Step "src/08_evaluate.py"; Run-Step "src/10_robustness_test.py"
if ($ComparePreprocessing) { Run-Step "src/03_image_processing.py"; Run-Step "src/09_compare_preprocessing.py" }
