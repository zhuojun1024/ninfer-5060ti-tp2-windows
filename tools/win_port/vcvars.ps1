# Import the MSVC x64 build environment into the current PowerShell session, which is what lets the
# other win_port scripts call cmake, ninja and nvcc directly instead of wrapping cmd.
#
# Usage:  . "$PSScriptRoot/vcvars.ps1"
#         . "$PSScriptRoot/vcvars.ps1" -VsPath "C:/Program Files/Microsoft Visual Studio/2022/Community"
#
# The Visual Studio installation is located in this order:
#   1. -VsPath, if given (the installation root, i.e. the directory that contains VC/).
#   2. $env:NINFER_VS_PATH, so the choice can live in the environment.
#   3. vswhere (shipped with every VS 2017+ installer), asking for the latest install that has the
#      MSVC x64 toolset.
#   4. The conventional per-edition paths under both Program Files roots.
param([string] $VsPath = $env:NINFER_VS_PATH)

$ErrorActionPreference = "Stop"

$relative = "VC/Auxiliary/Build/vcvars64.bat"
$candidates = [System.Collections.Generic.List[string]]::new()
if ($VsPath) { $candidates.Add((Join-Path $VsPath $relative)) }

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
if (Test-Path $vswhere) {
    $install = & $vswhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($install) { $candidates.Add((Join-Path $install.Trim() $relative)) }
}

foreach ($edition in @("Community", "Professional", "Enterprise", "BuildTools")) {
    foreach ($programFiles in @($env:ProgramFiles, ${env:ProgramFiles(x86)})) {
        if ($programFiles) {
            $candidates.Add((Join-Path $programFiles "Microsoft Visual Studio/2022/$edition/$relative"))
        }
    }
}

$vcvars = $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $vcvars) {
    throw ("vcvars64.bat not found. Pass -VsPath <VS installation root>, set NINFER_VS_PATH, or " +
           "install the Visual Studio C++ x64 build tools. Looked in: " + ($candidates -join "; "))
}

$capture = 'call "' + $vcvars + '" >nul 2>&1 && set'
cmd /c $capture | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2] }
}
if (-not $env:VCToolsInstallDir) { throw "importing the vcvars64 environment failed ($vcvars)" }
