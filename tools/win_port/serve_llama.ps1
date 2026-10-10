# Run a reference llama.cpp build for engine-to-engine comparison.
#
#   pwsh -File tools/win_port/serve_llama.ps1 -Root <llama.cpp build dir> -Model <gguf> `
#        -Mmproj <mmproj.gguf> -ChatTemplate <template>.jinja
#   pwsh -File tools/win_port/serve_llama.ps1            # start on 8080 with the tuned flags
#   pwsh -File tools/win_port/serve_llama.ps1 -Status    # listeners, health, VRAM
#   pwsh -File tools/win_port/serve_llama.ps1 -Stop      # stop every llama-server process
#
# The flag set is the owner's tuned ar3-opt recipe for this machine (tensor split over both 5060 Ti,
# q8_0/q5_0 KV, MTP speculative decoding). Never run this and serve.ps1 at the same time: both keep
# the 27B model resident in the two 16 GiB cards, and a second engine exhausts system memory too.
[CmdletBinding(DefaultParameterSetName = "start")]
param(
    [Parameter(ParameterSetName = "start")][switch] $Start,
    [Parameter(ParameterSetName = "stop")][switch] $Stop,
    [Parameter(ParameterSetName = "status")][switch] $Status,
    [string] $Root = $env:NINFER_LLAMA_ROOT,
    [string] $Model = $env:NINFER_LLAMA_MODEL,
    [string] $Mmproj = $env:NINFER_LLAMA_MMPROJ,
    [string] $ChatTemplate = $env:NINFER_LLAMA_CHAT_TEMPLATE,
    [int] $Port = 8080,
    [string] $LogFile = ""
)

$ErrorActionPreference = "Stop"

function Get-LlamaProcesses { @(Get-Process -Name llama-server -ErrorAction SilentlyContinue) }

if ($Stop) {
    Get-LlamaProcesses | Stop-Process -Force
    Start-Sleep -Seconds 3
    Write-Host ("stopped; remaining: " + (Get-LlamaProcesses).Count)
    & nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0
}

if ($Status) {
    Write-Host ("llama-server processes: " + (Get-LlamaProcesses).Count)
    Get-LlamaProcesses | ForEach-Object { Write-Host ("  pid " + $_.Id + "  started " + $_.StartTime) }
    try { Write-Host ("health: " + (Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5).Content) }
    catch { Write-Host "health: unreachable" }
    & nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
    exit 0
}

if ((Get-LlamaProcesses).Count -ne 0) { throw "a llama-server process is already running; stop it first (-Stop)" }
if (-not $Root) { throw "no llama.cpp build directory given: pass -Root <dir> or set NINFER_LLAMA_ROOT" }
foreach ($path in @($Model, $Mmproj, $ChatTemplate)) {
    if (-not $path) { throw "missing input: pass the artifact paths or set the NINFER_LLAMA_* variables" }
    if (-not (Test-Path $path)) { throw "missing input: $path" }
}
$binary = Join-Path $Root "llama-server.exe"
if (-not (Test-Path $binary)) { throw "llama-server.exe not found under $Root" }
if (-not $LogFile) { $LogFile = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "build-win/llama-serve.log" }

# The launcher loads ggml-cuda.dll and the CUDA runtime from its own directory.
$env:PATH = "$Root;$env:PATH"

$arguments = @(
    "-ngl", "99",
    "--temp", "0.7", "--top-k", "20", "--top-p", "0.80",
    "-c", "262144",
    "--spec-type", "draft-mtp", "--spec-draft-p-min", "0.1",
    "-fa", "on", "-ctk", "q8_0", "-ctv", "q5_0",
    "-sm", "tensor", "-ts", "1,1",
    "--jinja", "--chat-template-file", $ChatTemplate,
    "--reasoning-effort", "medium",
    "--host", "0.0.0.0", "--port", "$Port",
    "-mm", $Mmproj,
    "--spec-draft-n-max", "3",
    "-a", "Qwen3.8-27B-iMatrix-NVFP4-MTP",
    "-m", $Model,
    "-np", "3",
    "-dev", "CUDA0,CUDA1",
    "-lm", "none",
    "-devd", "CUDA0,CUDA1",
    "-kvu",
    "--repeat-penalty", "1.05"
)

Write-Host "running $binary (port $Port); Ctrl+C stops it"
& $binary @arguments 2>&1 | Tee-Object -FilePath $LogFile
exit $LASTEXITCODE
