#requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExpectedCommit,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string]$ExpectedTree,

    [Parameter(Mandatory = $true)]
    [string]$RepositoryRoot,

    [Parameter(Mandatory = $true)]
    [string]$RunRoot,

    [Parameter(Mandatory = $true)]
    [string]$EvidenceRoot,

    [Parameter(Mandatory = $true)]
    [string]$ResultParent,

    [Parameter(Mandatory = $true)]
    [string]$PythonExecutable,

    [Parameter(Mandatory = $true)]
    [string]$TempRoot,

    [Parameter(Mandatory = $true)]
    [string]$RecoveryStatePath,

    [switch]$ResumeOrStart,
    [switch]$RunAll
)

if ($MyInvocation.InvocationName -eq '.') {
    throw 'run_p1_analysis_campaign.ps1 must be executed, not dot-sourced'
}

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not $ResumeOrStart -or -not $RunAll) {
    throw 'both -ResumeOrStart and -RunAll are required'
}

$repository = [System.IO.Path]::GetFullPath($RepositoryRoot)
$run = [System.IO.Path]::GetFullPath($RunRoot)
$evidence = [System.IO.Path]::GetFullPath($EvidenceRoot)
$result = [System.IO.Path]::GetFullPath($ResultParent)
$temp = [System.IO.Path]::GetFullPath($TempRoot)
$python = [System.IO.Path]::GetFullPath($PythonExecutable)
$recoveryState = [System.IO.Path]::GetFullPath($RecoveryStatePath)

foreach ($path in @($repository, $run, $evidence, $python, $recoveryState)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "required path does not exist: $path"
    }
}
[System.IO.Directory]::CreateDirectory($result) | Out-Null
[System.IO.Directory]::CreateDirectory($temp) | Out-Null

$actualCommit = (& git -C $repository rev-parse 'HEAD^{commit}').Trim()
if ($LASTEXITCODE -ne 0 -or $actualCommit -ne $ExpectedCommit) {
    throw 'BLOCKED_SOURCE_IDENTITY: checkout commit mismatch'
}
$actualTree = (& git -C $repository rev-parse 'HEAD^{tree}').Trim()
if ($LASTEXITCODE -ne 0 -or $actualTree -ne $ExpectedTree) {
    throw 'BLOCKED_SOURCE_IDENTITY: checkout tree mismatch'
}
& git -C $repository symbolic-ref -q HEAD *> $null
if ($LASTEXITCODE -eq 0) {
    throw 'BLOCKED_SOURCE_IDENTITY: campaign checkout must be detached'
}
& git -C $repository diff --quiet --exit-code
if ($LASTEXITCODE -ne 0) {
    throw 'BLOCKED_SOURCE_IDENTITY: checkout has unstaged content changes'
}
& git -C $repository diff --cached --quiet --exit-code
if ($LASTEXITCODE -ne 0) {
    throw 'BLOCKED_SOURCE_IDENTITY: checkout has staged content changes'
}

$env:LC_ALL = 'C'
$env:TZ = 'UTC'
$env:PYTHONHASHSEED = '0'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CUDA_VISIBLE_DEVICES = ''
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$env:TEMP = $temp
$env:TMP = $temp
$env:TMPDIR = $temp
$env:PYTHONPYCACHEPREFIX = Join-Path $temp 'pycache'

$entrypoint = Join-Path $repository 'scripts\run_p1_analysis_campaign.py'
& $python -B $entrypoint `
    --expected-commit $ExpectedCommit `
    --expected-tree $ExpectedTree `
    --repository-root $repository `
    --run-root $run `
    --evidence-root $evidence `
    --result-parent $result `
    --config (Join-Path $repository 'configs\analysis\p1_analysis_campaign_v1.json') `
    --recovery-state $recoveryState `
    --platform-mode windows_local `
    --remote-branch origin/codex/vea-p1-t004a-local-r2 `
    --resume-or-start `
    --run-all
exit $LASTEXITCODE
