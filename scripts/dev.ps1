param(
    [ValidateSet('Setup', 'Check')]
    [string]$Task = 'Check',
    [string]$Python
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit $LASTEXITCODE): $Executable"
    }
}

Push-Location (Split-Path $PSScriptRoot -Parent)
$originalTemp = $env:TEMP
$originalTmp = $env:TMP
try {
    $taskTemp = Join-Path (Get-Location) ('artifacts/dev-tmp/' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $taskTemp -Force | Out-Null
    $env:TEMP = $taskTemp
    $env:TMP = $taskTemp
    $venvPython = Join-Path (Get-Location) '.venv/Scripts/python.exe'
    $versionCheck = 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "Expected Python 3.12")'
    if ($Task -eq 'Setup') {
        if (-not (Test-Path -LiteralPath $venvPython)) {
            if (Test-Path -LiteralPath '.venv') {
                throw 'Incomplete .venv exists; inspect it or move it aside before setup.'
            }
            if ($Python) {
                Invoke-Checked $Python @('-c', $versionCheck)
                Invoke-Checked $Python @('-m', 'venv', '.venv')
            } else {
                Invoke-Checked 'py' @('-3.12', '-m', 'venv', '.venv')
            }
        }
        Invoke-Checked $venvPython @('-c', $versionCheck)
        Invoke-Checked $venvPython @('-m', 'ensurepip')
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '-r', 'requirements-cpu.lock')
        Invoke-Checked $venvPython @('-m', 'pip', 'install', '-e', '.', '--no-deps', '--no-build-isolation')
        Invoke-Checked $venvPython @('-m', 'pip', 'check')
    } else {
        if (-not (Test-Path -LiteralPath $venvPython)) {
            throw 'Run scripts/dev.ps1 -Task Setup first.'
        }
        Invoke-Checked $venvPython @('-c', $versionCheck)
        Invoke-Checked $venvPython @('-m', 'pip', 'check')
        Invoke-Checked $venvPython @('scripts/check_repository.py')
        Invoke-Checked $venvPython @('-m', 'pytest', '-q')
    }
} finally {
    $env:TEMP = $originalTemp
    $env:TMP = $originalTmp
    Pop-Location
}
