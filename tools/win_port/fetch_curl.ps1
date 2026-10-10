# Fetch an MSVC-linkable libcurl (headers + import library) for the Windows product targets.
# src/product/media_acquire uses CURLOPT_PROTOCOLS_STR, so the bundle must be libcurl >= 7.85.
#
#   pwsh -File tools/win_port/fetch_curl.ps1
#   pwsh -File tools/win_port/fetch_curl.ps1 -Destination D:/deps -Proxy http://127.0.0.1:8080
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
$page = Join-Path $dst "index.html"
function Get-Via($url, $out) {
    try { Invoke-WebRequest -Uri $url -OutFile $out -TimeoutSec 120 }
    catch {
        if (-not $Proxy) { throw }
        Write-Output "direct failed ($($_.Exception.Message)); retrying through $Proxy"
        Invoke-WebRequest -Uri $url -OutFile $out -Proxy $Proxy -TimeoutSec 600
    }
}
if (-not (Test-Path $page)) { Get-Via "https://curl.se/windows/" $page }
$html = Get-Content $page -Raw
$href = [regex]::Match($html, 'href="([^"]*win64-mingw\.zip)"').Groups[1].Value
if (-not $href) { throw "no win64-mingw.zip link found on the curl.se/windows page" }
if ($href -notmatch '^https?://') { $href = "https://curl.se/windows/" + $href.TrimStart('.', '/') }
Write-Output "url=$href"
$zip = Join-Path $dst ([System.IO.Path]::GetFileName($href))
if (-not (Test-Path $zip)) { Get-Via $href $zip }
Get-Item $zip | Select-Object -ExpandProperty Length
$expanded = Join-Path $dst "expanded"
if (-not (Test-Path $expanded)) { Expand-Archive -Path $zip -DestinationPath $expanded -Force }
$root = Get-ChildItem $expanded -Directory | Select-Object -First 1
Get-ChildItem $root.FullName -Recurse -Include "curl.h","*.lib","*.dll.a","libcurl.dll" |
    ForEach-Object { $_.FullName.Replace($root.FullName, "") }
Write-Output ""
Write-Output "prefix:  $($root.FullName)"
Write-Output "use it:  `$env:NINFER_LIBCURL_ROOT = '$($root.FullName)'"
