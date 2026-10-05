param(
    [string]$Data = 'data/raw/UTKFace',
    [string]$Manifest = 'data/processed/manifest.csv',
    [string]$Config = 'configs/default.json',
    [string]$RunPrefix = 'experiment01',
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) { throw 'Buat .venv dan pasang requirements terlebih dahulu.' }
if ($RunPrefix -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunPrefix hanya boleh memuat huruf, angka, underscore, dan tanda minus.' }
foreach ($method in @('b0','b1','m1','b1_control')) {
    $arguments = @('-m','faceage','train','--method',$method,'--data',$Data,'--manifest',$Manifest,
                   '--config',$Config,'--device',$Device,'--output',"runs/${RunPrefix}_${method}")
    if ($method -in @('m1','b1_control')) { $arguments += @('--init-checkpoint',"runs/${RunPrefix}_b1/best.pt") }
    & $projectPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "Pelatihan $method gagal. Periksa pesan dan pilih nama run baru bila mengulang." }
}
& $projectPython -m faceage compare --data $Data --manifest $Manifest --device $Device `
    --runs "runs/${RunPrefix}_b0" "runs/${RunPrefix}_b1" "runs/${RunPrefix}_m1" "runs/${RunPrefix}_b1_control" `
    --output "runs/${RunPrefix}_comparison"
if ($LASTEXITCODE -ne 0) { throw 'Perbandingan validasi gagal.' }
Write-Output 'Pelatihan dan perbandingan validasi selesai. Gunakan --split test --final-test setelah konfigurasi dibekukan.'
