# Package a self-contained Windows release archive from a finished build.
#
#   pwsh -File tools/win_port/package.ps1                 # version read from the built executables
#   pwsh -File tools/win_port/package.ps1 -RequireTag      # release build: HEAD must carry v<version>
#   pwsh -File tools/win_port/package.ps1 -OutDir dist     # where the archive is written
#
# The archive is <OutDir>/ninfer-<version>-win-x64.zip. It holds bin/ with the three executables and
# every non-system DLL they load, so a user needs no CUDA toolkit, FFmpeg or curl installation - only
# an NVIDIA display driver, which owns nvcuda.dll. LICENSE, a generated README.txt and the bundled
# third-party license texts sit next to bin/.
#
# Publishing stays manual, on purpose: this script never tags, pushes or uploads. Build, package,
# tag the commit as v<version>, then attach the archive to the GitHub release.
#
# The version has one source of truth - the executables themselves. They are built with
# NINFER_VERSION from project() in CMakeLists.txt, so the script refuses to package a tree whose
# executables disagree or whose version does not match the tag it is asked to release.
[CmdletBinding()]
param(
    [switch] $RequireTag,
    [string] $BuildDir = "",
    [string] $AppsDir = "",
    [string] $OutDir = "",
    [string] $FfmpegRoot = $env:NINFER_FFMPEG_ROOT,
    [string] $LibcurlRoot = $env:NINFER_LIBCURL_ROOT,
    [string] $CudaRoot = $env:CUDA_PATH,
    [switch] $KeepStaging
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/vcvars.ps1"

$SourceDir = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $BuildDir) { $BuildDir = Join-Path $SourceDir "build-win" }
if (-not $AppsDir) { $AppsDir = Join-Path $BuildDir "apps" }
if (-not $OutDir) { $OutDir = Join-Path $SourceDir "dist" }

foreach ($prefix in @(
        @{ Name = "FFmpeg"; Value = $FfmpegRoot; Flag = "-FfmpegRoot"; Env = "NINFER_FFMPEG_ROOT" },
        @{ Name = "libcurl"; Value = $LibcurlRoot; Flag = "-LibcurlRoot"; Env = "NINFER_LIBCURL_ROOT" },
        @{ Name = "CUDA"; Value = $CudaRoot; Flag = "-CudaRoot"; Env = "CUDA_PATH" })) {
    if (-not $prefix.Value) {
        throw ("{0} prefix not set: pass {1} <dir> or set {2}" -f $prefix.Name, $prefix.Flag, $prefix.Env)
    }
    if (-not (Test-Path $prefix.Value)) {
        throw ("{0} prefix not found: {1}" -f $prefix.Name, $prefix.Value)
    }
}
if (-not (Test-Path $AppsDir)) {
    throw "built executables not found: $AppsDir (run tools/win_port/build.ps1 first)"
}

# Quotes an argument list the way Start-Process needs it: it joins the array with spaces and does not
# quote elements, so any element with a space would otherwise split into two arguments.
function Format-ArgumentList([string[]] $Arguments) {
    return (($Arguments | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } }) -join " ")
}

# Runs one native tool with its streams bound to files (never a pipeline, for the reason documented in
# build.ps1) and returns its exit code with the captured lines.
function Invoke-Captured([string] $Exe, [string[]] $Arguments, [string] $WorkingDirectory) {
    $out = Join-Path $env:TEMP ("ninfer-package-" + [guid]::NewGuid().ToString("N") + ".out")
    $err = "$out.err"
    $process = Start-Process -FilePath $Exe -ArgumentList (Format-ArgumentList $Arguments) `
        -WorkingDirectory $WorkingDirectory -NoNewWindow -PassThru `
        -RedirectStandardOutput $out -RedirectStandardError $err
    if (-not $process.WaitForExit(600000)) {
        $process.Kill()
        $process.WaitForExit()
        throw ("{0} did not finish within 600 s" -f $Exe)
    }
    # Both arrays are pre-typed: a bare assignment from an if-statement would collapse a one-line
    # capture into a string, and indexing that returns a character rather than the line.
    [string[]] $lines = @()
    [string[]] $errors = @()
    if (Test-Path $out) { $lines = [System.IO.File]::ReadAllLines($out) }
    if (Test-Path $err) { $errors = [System.IO.File]::ReadAllLines($err) }
    Remove-Item $out, $err -ErrorAction SilentlyContinue
    return [pscustomobject]@{ ExitCode = $process.ExitCode; Lines = $lines; ErrorLines = $errors }
}

function Get-GitLines([string[]] $Arguments) {
    $result = Invoke-Captured "git" $Arguments $SourceDir
    if ($result.ExitCode -ne 0) { throw ("git " + ($Arguments -join " ") + " failed: " + ($result.ErrorLines -join " ")) }
    return $result.Lines
}

# The executables are the authority on the version: they print what NINFER_VERSION compiled into them.
$products = @("ninfer", "ninfer-serve", "ninfer-perplexity")
$runtimeDirs = @($AppsDir,
    (Join-Path $FfmpegRoot "bin"),
    (Join-Path $LibcurlRoot "bin"),
    (Join-Path $CudaRoot "bin\x64"),
    (Join-Path $CudaRoot "bin")) | Where-Object { Test-Path $_ }
$env:PATH = (($runtimeDirs -join ";") + ";" + $env:PATH)

$version = $null
foreach ($product in $products) {
    $exe = Join-Path $AppsDir "$product.exe"
    if (-not (Test-Path $exe)) { throw "missing executable: $exe (run tools/win_port/build.ps1 first)" }
    $result = Invoke-Captured $exe @("--version") $AppsDir
    $reported = if ($result.Lines.Count -eq 1) { ($result.Lines[0] -split "\s+")[-1] } else { "" }
    if ($result.ExitCode -ne 0 -or $reported -notmatch '^[0-9]+\.[0-9]+\.[0-9]+') {
        throw ("{0} --version did not report a release version (exit {1}): {2}" -f $product,
            $result.ExitCode, ($result.Lines + $result.ErrorLines) -join " ")
    }
    if (-not $version) { $version = $reported }
    elseif ($version -ne $reported) {
        throw ("the executables disagree about the version: {0} vs {1} ({2})" -f $version, $reported, $product)
    }
}
Write-Host ("package: version {0} (from the built executables)" -f $version)

$tag = "v$version"
$tagsAtHead = @(Get-GitLines @("tag", "--points-at", "HEAD"))
$dirty = @(Get-GitLines @("status", "--porcelain"))
if ($tagsAtHead -contains $tag) {
    Write-Host ("package: HEAD carries tag {0}" -f $tag)
} elseif ($RequireTag) {
    throw ("HEAD does not carry tag {0} (tags at HEAD: {1}); tag the release commit first" -f
        $tag, $(if ($tagsAtHead.Count -eq 0) { "none" } else { $tagsAtHead -join ", " }))
} else {
    Write-Host ("package: warning: HEAD does not carry tag {0}; tag it before publishing" -f $tag)
}
if ($dirty.Count -ne 0) {
    Write-Host ("package: warning: the working tree has {0} uncommitted change(s)" -f $dirty.Count)
}

# Resolve the DLL closure the same way Windows does, except that vendor directories win over the
# system directory: msvcp140.dll is in System32 on a machine with the VC redist installed, but a
# release has to carry the redistributable copy so it also runs where the redist is absent.
$vendorDirs = @($AppsDir,
    (Join-Path $FfmpegRoot "bin"),
    (Join-Path $LibcurlRoot "bin"),
    (Join-Path $CudaRoot "bin\x64"),
    (Join-Path $CudaRoot "bin"),
    (Join-Path $env:VCToolsRedistDir "x64/Microsoft.VC143.CRT"),
    (Join-Path $env:VCToolsRedistDir "x64")) | Where-Object { $_ -and (Test-Path $_) }
$systemDir = Join-Path $env:SystemRoot "System32"
$systemDirs = @($systemDir, (Join-Path $systemDir "downlevel")) | Where-Object { Test-Path $_ }

$bundled = [ordered]@{}
$queue = [System.Collections.Generic.Queue[string]]::new()
foreach ($product in $products) { $queue.Enqueue((Join-Path $AppsDir "$product.exe")) }
while ($queue.Count -ne 0) {
    $image = $queue.Dequeue()
    $result = Invoke-Captured "dumpbin" @("/nologo", "/dependents", (Split-Path -Leaf $image)) (Split-Path -Parent $image)
    if ($result.ExitCode -ne 0) {
        throw ("dumpbin failed on {0}: {1}" -f $image, ($result.ErrorLines -join " "))
    }
    # A dependency line is a bare file name. Requiring "no whitespace" drops dumpbin's
    # "Dump of file <name>.dll" header, which also ends in .dll.
    $dependencies = @($result.Lines | ForEach-Object { $_.Trim() } | Where-Object { $_ -match '^[^\\/\s]+\.dll$' })
    foreach ($name in $dependencies) {
        # api-ms-win-* and ext-ms-win-* are OS API sets. Windows resolves them through the API set
        # schema and they must never be shipped; on this machine they are not even files in System32,
        # so they are skipped before any lookup.
        if ($name -match '^(api-ms-win|ext-ms-win)-') { continue }
        if ($bundled.Contains($name)) { continue }
        $source = $null
        foreach ($dir in $vendorDirs) {
            $candidate = Join-Path $dir $name
            if (Test-Path $candidate) { $source = $candidate; break }
        }
        if ($source) {
            $bundled[$name] = $source
            $queue.Enqueue($source)
        } elseif (-not ($systemDirs | Where-Object { Test-Path (Join-Path $_ $name) })) {
            throw ("cannot locate {0}, which {1} loads; searched: {2}" -f $name, (Split-Path -Leaf $image),
                ($vendorDirs -join "; "))
        }
    }
}
if ($bundled.Count -eq 0) { throw "no DLL dependencies were resolved; is the build complete?" }
$bundledMiB = [math]::Round((($bundled.Values | ForEach-Object { (Get-Item $_).Length } | Measure-Object -Sum).Sum) / 1MB, 1)
Write-Host ("package: {0} DLLs, {1} MiB" -f $bundled.Count, $bundledMiB)

$name = "ninfer-$version-win-x64"
$staging = Join-Path $OutDir $name
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
$binDir = Join-Path $staging "bin"
$licenseDir = Join-Path $staging "licenses"
$null = New-Item -ItemType Directory -Force -Path $binDir, $licenseDir

foreach ($product in $products) { Copy-Item (Join-Path $AppsDir "$product.exe") $binDir }
foreach ($entry in $bundled.GetEnumerator()) { Copy-Item $entry.Value $binDir }
Copy-Item (Join-Path $SourceDir "LICENSE") (Join-Path $staging "LICENSE")

# License texts travel with the binaries they cover. The MSVC runtime is deployed app-local under the
# Visual Studio license terms and ships no redistributable text of its own.
$licenseFiles = [ordered]@{
    "cuda-LICENSE.txt"   = (Join-Path $CudaRoot "LICENSE")
    "ffmpeg-LICENSE.txt" = (Join-Path $FfmpegRoot "LICENSE.txt")
    "curl-COPYING.txt"   = (Join-Path $LibcurlRoot "COPYING.txt")
}
$copiedLicenses = [ordered]@{}
foreach ($entry in $licenseFiles.GetEnumerator()) {
    if (Test-Path $entry.Value) {
        Copy-Item $entry.Value (Join-Path $licenseDir $entry.Key)
        $copiedLicenses[$entry.Key] = $entry.Value
    }
}

# The FFmpeg build identifies itself, and that identifier is what a GPL recipient needs in order to
# fetch the corresponding source, so record it verbatim instead of hard-coding one distributor's build.
$ffmpegBuild = "the FFmpeg build shipped in this archive"
$ffmpegExe = Join-Path $FfmpegRoot "bin/ffmpeg.exe"
if (Test-Path $ffmpegExe) {
    $probe = Invoke-Captured $ffmpegExe @("-version") $AppsDir
    if ($probe.Lines.Count -ne 0 -and $probe.Lines[0] -match '^ffmpeg version (\S+)') {
        $ffmpegBuild = $Matches[1]
    }
}

# The notices are read by the people who receive the archive, so they list components and sizes
# without this machine's paths; the resolution is echoed on the console for the maintainer instead.
$dllLines = ($bundled.GetEnumerator() | ForEach-Object {
        "  {0}  {1} MiB" -f $_.Key, [math]::Round((Get-Item $_.Value).Length / 1MB, 1)
    }) -join "`r`n"
foreach ($entry in $bundled.GetEnumerator()) {
    Write-Host ("package:   {0} <- {1}" -f $entry.Key, $entry.Value)
}

$notices = @'
NInfer __VERSION__ for Windows x64 - bundled third-party components
===================================================================

NInfer itself is licensed under the Apache License 2.0; see LICENSE.

bin/ holds the NInfer executables and every non-system DLL they load:

__DLLS__

NVIDIA CUDA Toolkit - cuBLAS (cublas64_13.dll, cublasLt64_13.dll)
  Redistributable under the CUDA Toolkit End User License Agreement, whose Attachment A
  ("Distributable Software") lists the cuBLAS libraries and permits the version-bearing file names
  used here. License text: licenses/cuda-LICENSE.txt.
  Bundling them is what lets this archive run without a local CUDA toolkit installation; the NVIDIA
  display driver, which provides nvcuda.dll, is still required.

FFmpeg (avcodec-63.dll, avformat-63.dll, avutil-61.dll, swresample-7.dll, swscale-10.dll)
  ffmpeg version __FFMPEG__ - licensed under the GNU General Public License version 3, because this
  build was configured with --enable-gpl --enable-version3. The libraries are distributed unmodified
  and are loaded at run time as separate shared objects. License text: licenses/ffmpeg-LICENSE.txt.
  Corresponding source: the FFmpeg revision named above, available from https://ffmpeg.org/download.html
  (git https://git.ffmpeg.org/ffmpeg.git).

curl (libcurl-x64.dll)
  Licensed under the curl license; see licenses/curl-COPYING.txt.

Microsoft Visual C++ runtime (msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll)
  Deployed app-local as permitted by the Microsoft Visual Studio license terms.
'@
$notices = $notices.Replace("__VERSION__", $version).Replace("__DLLS__", $dllLines).Replace("__FFMPEG__", $ffmpegBuild)
[System.IO.File]::WriteAllText((Join-Path $staging "THIRD-PARTY-NOTICES.txt"), $notices, [System.Text.UTF8Encoding]::new($false))

$readme = @'
NInfer __VERSION__ for Windows x64 - dual-GPU TP-2 build
========================================================

  bin\ninfer-serve.exe       OpenAI- and Anthropic-compatible server (the product route)
  bin\ninfer.exe             one-shot CLI generation
  bin\ninfer-perplexity.exe  offline perplexity and CausalScoring
  bin\*.dll                  CUDA, FFmpeg, curl and MSVC runtime libraries the executables load

Requirements
  * Two RTX 5060 Ti (16 GiB) cards for the TP-2 route; a single card also runs.
  * An NVIDIA display driver new enough for CUDA 13.3 (nvcuda.dll comes from the driver).
  * A .ninfer artifact. Weights are not part of this archive.

Serve on both cards (DFlash2 with seven draft tokens is the fastest measured route):

  bin\ninfer-serve.exe <artifact>.ninfer --devices 0,1 --max-context 131072 --kv-dtype fp8 ^
      --port 8099 --spec dflash2 --draft-tokens 7 --lm-head-draft

Add --vision when the artifact carries a vision component. Then post to
http://127.0.0.1:8099/v1/chat/completions. The repository README covers the plain, MTP, multi-request
and long-context-reuse configurations:

  https://github.com/zhuojun1024/ninfer-5060ti-tp2-windows

NInfer is Apache-2.0 (see LICENSE). The bundled libraries and their licenses are listed in
THIRD-PARTY-NOTICES.txt, with the license texts under licenses/.
'@
$readme = $readme.Replace("__VERSION__", $version)
[System.IO.File]::WriteAllText((Join-Path $staging "README.txt"), $readme, [System.Text.UTF8Encoding]::new($false))

$zip = Join-Path $OutDir "$name.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
$started = Get-Date
Write-Host ("package: compressing {0} ..." -f $name)
Compress-Archive -Path $staging -DestinationPath $zip -CompressionLevel Optimal
$seconds = [int]((Get-Date) - $started).TotalSeconds
if (-not $KeepStaging) { Remove-Item $staging -Recurse -Force }

$archive = Get-Item $zip
$hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
Write-Host ""
Write-Host ("archive: {0}" -f $archive.FullName)
Write-Host ("size:    {0} MiB ({1} s)" -f [math]::Round($archive.Length / 1MB, 1), $seconds)
Write-Host ("sha256:  {0}" -f $hash)
Write-Host ("licenses: {0}" -f (($copiedLicenses.Keys) -join ", "))
Write-Host ""
Write-Host "publish (manual, from this repository):"
if ($tagsAtHead -contains $tag) {
    Write-Host ("  tag {0} already exists at HEAD; push it with the branch" -f $tag)
} else {
    Write-Host ("  git tag -a {0} -m `"NInfer {0}`"" -f $tag)
}
Write-Host "  git push fork main --follow-tags"
Write-Host ("  then draft a release for {0} on GitHub and attach {1}" -f $tag, $archive.Name)
Write-Host "PACKAGE_EXIT=0"
exit 0
