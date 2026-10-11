# Fetch an MSVC-linkable FFmpeg (headers + import libraries) for the native Windows build.
# A runtime-only FFmpeg tree has no include/ or lib/, so the core media decoder (src/media/decode)
# cannot be compiled against it. BtbN's shared win64 build ships both.
#
# The LGPL build is fetched by default: NInfer only calls FFmpeg's decoders, so nothing here needs the
# GPL-only parts, and a redistributed archive then carries no GPL source-offer obligation. A GPL
# prefix still builds and packages (package.ps1 writes the notice entry that matches its LICENSE.txt).
#
#   pwsh -File tools/win_port/fetch_ffmpeg.ps1
#   pwsh -File tools/win_port/fetch_ffmpeg.ps1 -Destination D:/deps -Proxy http://127.0.0.1:8080
#
# The script prints the prefix to hand to build.ps1; it also prints the environment variable that
# makes it the default. Network access is direct by default; -Proxy (or $env:NINFER_PROXY) is used
# only as a retry when the direct download fails.
[CmdletBinding()]
param(
    [string] $Destination = $(if ($env:NINFER_DEPS_DIR) { $env:NINFER_DEPS_DIR }
                              else { Join-Path $env:USERPROFILE "ninfer-deps" }),
    [string] $Proxy = $env:NINFER_PROXY
)

$ErrorActionPreference = "Stop"
$dst = $Destination
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip"
$out = Join-Path $dst "ffmpeg-win64-shared.zip"
if (-not (Test-Path $out)) {
    try {
        Write-Output "direct download"
        Invoke-WebRequest -Uri $url -OutFile $out -TimeoutSec 180
    } catch {
        if (-not $Proxy) { throw }
        Write-Output "direct failed ($($_.Exception.Message)); retrying through $Proxy"
        Invoke-WebRequest -Uri $url -OutFile $out -Proxy $Proxy -TimeoutSec 600
    }
}
Get-Item $out | Select-Object -ExpandProperty Length
$expanded = Join-Path $dst "expanded"
if (-not (Test-Path $expanded)) {
    Expand-Archive -Path $out -DestinationPath $expanded -Force
}
$root = Get-ChildItem $expanded -Directory | Select-Object -First 1
foreach ($p in @("include\libavcodec\avcodec.h", "include\libswscale\swscale.h",
                 "lib\avcodec.lib", "lib\avformat.lib", "lib\avutil.lib", "lib\swscale.lib")) {
    Write-Output "$([bool](Test-Path (Join-Path $root.FullName $p)))  $p"
}
Write-Output ""
Write-Output "prefix:  $($root.FullName)"
Write-Output "use it:  `$env:NINFER_FFMPEG_ROOT = '$($root.FullName)'"
