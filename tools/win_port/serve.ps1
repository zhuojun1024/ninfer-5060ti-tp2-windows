# Run the native Windows TP-2 server, or inspect/stop it.
#
#   pwsh -File tools/win_port/serve.ps1                  # start with the shipped recipe
#   pwsh -File tools/win_port/serve.ps1 -Context 32768   # larger KV capacity
#   pwsh -File tools/win_port/serve.ps1 -Spec dflash2 -DraftTokens 7   # DFlash2 masked draft
#   pwsh -File tools/win_port/serve.ps1 -Plain           # no MTP/vision, plain decode
#   pwsh -File tools/win_port/serve.ps1 -HostKvMiB 0     # no cross-session KV retention
#   pwsh -File tools/win_port/serve.ps1 -Model <artifact>.ninfer   # or set NINFER_MODEL
#   pwsh -File tools/win_port/serve.ps1 -ChatTemplate <template>.jinja   # custom chat template
#   pwsh -File tools/win_port/serve.ps1 -VisionItemTokens 4096   # smaller Vision workspace
#   pwsh -File tools/win_port/serve.ps1 -Status          # processes, health, VRAM
#   pwsh -File tools/win_port/serve.ps1 -Stop            # stop every ninfer-serve process
#
# Two 16 GiB cards hold the 27B NVFP4 artifact: 131072 tokens of KV leaves about 1.6 GiB free per shard
# on this machine, and 262144 (the recipe used under WSL) leaves too little for comfort under WDDM.
# Only ninfer-serve is managed here: no llama.cpp process is ever touched.
[CmdletBinding(DefaultParameterSetName = "start")]
param(
    [Parameter(ParameterSetName = "start")][switch] $Start,
    [Parameter(ParameterSetName = "stop")][switch] $Stop,
    [Parameter(ParameterSetName = "status")][switch] $Status,
    [string] $Model = $env:NINFER_MODEL,
    [string] $Binary = "",
    [int] $Port = 8099,
    [int] $Context = 131072,
    [ValidateSet("mtp", "dflash2")][string] $Spec = "mtp",
    [int] $DraftTokens = 2,
    [int] $HostKvMiB = 32768,
    [int] $PrivateContinuations = 6,
    [switch] $Plain,
    [switch] $Background,
    [string] $LogFile = "",
    [string] $FfmpegRoot = $env:NINFER_FFMPEG_ROOT,
    [string] $LibcurlRoot = $env:NINFER_LIBCURL_ROOT,
    [string] $ChatTemplate = "",
    [int] $VisionItemTokens = 16384
)

$ErrorActionPreference = "Stop"

function Get-ServeProcesses { @(Get-Process -Name ninfer-serve -ErrorAction SilentlyContinue) }

function Show-Status {
    $processes = Get-ServeProcesses
    Write-Host ("ninfer-serve processes: " + $processes.Count)
    foreach ($process in $processes) {
        Write-Host ("  pid " + $process.Id + "  started " + $process.StartTime)
    }
    try {
        $health = (Invoke-WebRequest -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 5).Content
        Write-Host "health: $health"
    } catch {
        Write-Host "health: unreachable"
    }
    & nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
}

if ($Stop) {
    Get-ServeProcesses | Stop-Process -Force
    Start-Sleep -Seconds 3
    Write-Host ("stopped; remaining: " + (Get-ServeProcesses).Count)
    & nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 0
}

if ($Status) { Show-Status; exit 0 }

if ((Get-ServeProcesses).Count -ne 0) {
    throw "a ninfer-serve process is already running; stop it first (-Stop) instead of holding two engines in VRAM"
}
if (-not $Model) { throw "no model artifact given: pass -Model <file.ninfer> or set NINFER_MODEL" }
if (-not (Test-Path $Model)) { throw "model artifact not found: $Model" }
if (-not $Binary) {
    $Binary = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "build-win/apps/ninfer-serve.exe"
}
if (-not (Test-Path $Binary)) { throw "server binary not found: $Binary (run tools/win_port/build.ps1 first)" }
if ($VisionItemTokens -lt 2048 -or $VisionItemTokens -gt 16384) { throw "VisionItemTokens must be in [2048,16384]" }
if ($ChatTemplate -and -not (Test-Path $ChatTemplate)) { throw "chat template not found: $ChatTemplate" }
if (-not $LogFile) { $LogFile = Join-Path (Split-Path -Parent $Binary) "serve-win.log" }

# The server loads avcodec/swscale and libcurl at run time, so both DLL directories must be visible.
if (-not (Test-Path $FfmpegRoot)) {
    throw ("FFmpeg prefix not found: '$FfmpegRoot' (pass -FfmpegRoot <dir> or set NINFER_FFMPEG_ROOT; " +
           "tools/win_port/fetch_ffmpeg.ps1 prints the value)")
}
if (-not (Test-Path $LibcurlRoot)) {
    throw ("libcurl prefix not found: '$LibcurlRoot' (pass -LibcurlRoot <dir> or set " +
           "NINFER_LIBCURL_ROOT; tools/win_port/fetch_curl.ps1 prints the value)")
}
$env:PATH = "$FfmpegRoot/bin;$LibcurlRoot/bin;$env:PATH"

$arguments = @(
    $Model,
    "--devices", "0,1",
    "--max-context", "$Context",
    "--port", "$Port",
    "--host", "127.0.0.1",
    "--kv-dtype", "fp8",
    # Cross-session KV retention: 32 GiB of pinned host KV (16 GiB/card = five 204,800-token fp8
    # conversations) with the default six-entry catalog. The arena is pinned on the first eviction,
    # so a single-conversation workload never pays for it; -HostKvMiB 0 turns retention off.
    "--host-kv-mib", "$HostKvMiB",
    "--max-private-continuations", "$PrivateContinuations",
    "--log-level", "info"
)
if ($ChatTemplate) { $arguments += @("--chat-template", $ChatTemplate) }
if ($VisionItemTokens -ne 16384) { $arguments += @("--vision-item-tokens", "$VisionItemTokens") }
if (-not $Plain) {
    $arguments += @(
        "--temperature", "0.7", "--top-k", "20", "--top-p", "0.80",
        "--spec", "$Spec", "--draft-tokens", "$DraftTokens", "--lm-head-draft",
        "--vision", "--reasoning-effort", "medium"
    )
}

if ($Background) {
    # A background child belongs to the launching shell's job object, so it dies when that shell is torn
    # down (an agent harness running this script inside one tool call kills it immediately). Use this only
    # from a terminal you are keeping open.
    # ninfer-serve writes its diagnostics (including the memory ledger) to stderr, so that stream owns the
    # main log file and stdout goes to a sidecar.
    $process = Start-Process -FilePath $Binary -ArgumentList $arguments -NoNewWindow -PassThru -RedirectStandardOutput "$LogFile.out" -RedirectStandardError $LogFile
    Write-Host ("pid " + $process.Id + " (background child of this shell); log " + $LogFile)
} else {
    # Foreground is the default: the shell that starts the server keeps ownership of it, the log streams
    # to the console, and Ctrl+C stops it.
    Write-Host "running $Binary (context $Context); Ctrl+C stops it"
    # ninfer-serve diagnostics (including the memory ledger) all go to stderr. Merging them into the
    # success stream with 2>&1 makes PowerShell wrap every line in an ErrorRecord, which the console
    # then renders in red regardless of log level. Casting to string before Tee keeps the colour
    # consistent with the content.
    & $Binary @arguments 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $LogFile
    exit $LASTEXITCODE
}
