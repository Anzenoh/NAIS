[CmdletBinding()]
param(
    [string]$ProjectRoot = '',
    [string]$StarterKitZip = '',
    [switch]$SkipStarterKit,
    [switch]$SkipGitInit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Default to the directory containing this script. This avoids source-file encoding
# problems with non-ASCII absolute paths in Windows PowerShell 5.1.
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = $PSScriptRoot
}
$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)

if ([string]::IsNullOrWhiteSpace($StarterKitZip)) {
    $StarterKitZip = Join-Path $PSScriptRoot 'vision_encoder_alignment_codex_kit.zip'
}

$CreatedItems = [System.Collections.Generic.List[string]]::new()
$SkippedItems = [System.Collections.Generic.List[string]]::new()
$Warnings = [System.Collections.Generic.List[string]]::new()

function New-DirectoryIfMissing {
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
        $CreatedItems.Add($Path)
    }
    else {
        $SkippedItems.Add($Path)
    }
}

function New-FileIfMissing {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][AllowEmptyString()][string]$Content
    )

    $Parent = Split-Path -Parent $Path
    if ($Parent) {
        New-DirectoryIfMissing -Path $Parent
    }

    if (-not (Test-Path -LiteralPath $Path)) {
        Set-Content -LiteralPath $Path -Value $Content -Encoding utf8
        $CreatedItems.Add($Path)
    }
    else {
        $SkippedItems.Add($Path)
    }
}

function Copy-TreeWithoutOverwrite {
    param(
        [Parameter(Mandatory)][string]$SourceRoot,
        [Parameter(Mandatory)][string]$DestinationRoot
    )

    Get-ChildItem -LiteralPath $SourceRoot -Recurse -Force | ForEach-Object {
        $RelativePath = $_.FullName.Substring($SourceRoot.Length).TrimStart([char[]]'\/')
        $DestinationPath = Join-Path $DestinationRoot $RelativePath

        if ($_.PSIsContainer) {
            New-DirectoryIfMissing -Path $DestinationPath
        }
        else {
            $DestinationParent = Split-Path -Parent $DestinationPath
            New-DirectoryIfMissing -Path $DestinationParent

            if (-not (Test-Path -LiteralPath $DestinationPath)) {
                Copy-Item -LiteralPath $_.FullName -Destination $DestinationPath
                $CreatedItems.Add($DestinationPath)
            }
            else {
                $SkippedItems.Add($DestinationPath)
            }
        }
    }
}

function New-PythonStubIfMissing {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Purpose
    )

    $Content = @"
"""$Purpose

Implementation is intentionally deferred to a reviewed Codex task.
"""
"@
    New-FileIfMissing -Path $Path -Content $Content
}

Write-Host "[1/7] Validate destination" -ForegroundColor Cyan
if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    $ParentPath = Split-Path -Parent $ProjectRoot
    if (-not $ParentPath -or -not (Test-Path -LiteralPath $ParentPath)) {
        throw "Project root and its parent do not exist: $ProjectRoot"
    }
}

New-DirectoryIfMissing -Path $ProjectRoot

Write-Host "[2/7] Install Codex starter documents when available" -ForegroundColor Cyan
if (-not $SkipStarterKit) {
    if (Test-Path -LiteralPath $StarterKitZip) {
        $TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("vea_starter_" + [Guid]::NewGuid().ToString('N'))
        try {
            New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
            Expand-Archive -LiteralPath $StarterKitZip -DestinationPath $TempRoot -Force

            $ExtractedDirectories = @(Get-ChildItem -LiteralPath $TempRoot -Directory)
            if ($ExtractedDirectories.Count -eq 1) {
                $StarterRoot = $ExtractedDirectories[0].FullName
            }
            else {
                $StarterRoot = $TempRoot
            }

            Copy-TreeWithoutOverwrite -SourceRoot $StarterRoot -DestinationRoot $ProjectRoot
        }
        finally {
            if (Test-Path -LiteralPath $TempRoot) {
                Remove-Item -LiteralPath $TempRoot -Recurse -Force
            }
        }
    }
    else {
        $Warnings.Add("Starter kit ZIP not found: $StarterKitZip. Minimal governance files will be created instead.")
    }
}

Write-Host "[3/7] Create repository directories" -ForegroundColor Cyan
$Directories = @(
    'configs',
    'data',
    'data\manifests',
    'src',
    'src\vision_encoder_alignment',
    'src\vision_encoder_alignment\extractors',
    'src\vision_encoder_alignment\metrics',
    'src\vision_encoder_alignment\utils',
    'scripts',
    'tests',
    'tests\unit',
    'tests\integration',
    'artifacts',
    'artifacts\features',
    'artifacts\metrics',
    'artifacts\figures',
    'docs',
    'docs\contracts',
    'docs\templates',
    'tasks',
    'reports',
    'reports\runs'
)

foreach ($Directory in $Directories) {
    New-DirectoryIfMissing -Path (Join-Path $ProjectRoot $Directory)
}

Write-Host "[4/7] Create non-destructive skeleton files" -ForegroundColor Cyan
$MinimalReadme = @'
# Vision Encoder Alignment

Research repository for analyzing and aligning hierarchical representations across heterogeneous vision encoders.

The initial phases are encoder-only. LLM integration is outside the core representation-analysis scope unless a later task explicitly adds it.
'@
New-FileIfMissing -Path (Join-Path $ProjectRoot 'README.md') -Content $MinimalReadme

$MinimalAgents = @'
# AGENTS.md

Before changing code, read `docs/PROJECT_CONTEXT.md`, `docs/RESEARCH_CONTRACT.md`, and the active task file under `tasks/`.

Core constraints:
- Keep vision encoders frozen during the initial representation-analysis phases.
- Do not mix raw backbone features with post-projector or LLM-aligned features.
- Split train, validation, and test data by image, never by patch.
- Do not download models or datasets unless the active task explicitly permits it.
- Do not commit, push, or create remotes unless explicitly requested.
- Report commands actually executed and never claim unexecuted validation as PASS.
'@
New-FileIfMissing -Path (Join-Path $ProjectRoot 'AGENTS.md') -Content $MinimalAgents

$GitIgnore = @'
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# Virtual environments
.venv/
venv/
env/

# IDE / OS
.vscode/
.idea/
.DS_Store
Thumbs.db

# Secrets
.env
.env.*
*.pem
*.key

# Model and dataset caches
.cache/
checkpoints/
models/
data/raw/
data/downloads/

# Generated research artifacts
artifacts/**
!artifacts/
!artifacts/.gitkeep
!artifacts/features/
!artifacts/features/.gitkeep
!artifacts/metrics/
!artifacts/metrics/.gitkeep
!artifacts/figures/
!artifacts/figures/.gitkeep

# Temporary outputs
*.tmp
*.log
'@
New-FileIfMissing -Path (Join-Path $ProjectRoot '.gitignore') -Content $GitIgnore

$ExperimentConfig = @'
experiment:
  name: canonical_224_encoder_alignment
  stage: smoke
  seed: 42

input:
  image_size: [224, 224]
  geometry_policy: shared_resize_and_center_crop
  model_native_normalization: true

features:
  include_special_tokens: false
  global_pool: patch_mean
  selected_depth_fractions: [0.25, 0.5, 0.75, 1.0]
  storage_dtype: float16

runtime:
  device: cpu
  batch_size: 1
  num_workers: 0
  network_access: false
'@
New-FileIfMissing -Path (Join-Path $ProjectRoot 'configs\experiment_canonical_224.yaml') -Content $ExperimentConfig

$ModelConfig = @'
models:
  clip_vit_b16:
    checkpoint: openai/clip-vit-base-patch16
    revision: null
    enabled: true

  siglip2_base_patch16_224:
    checkpoint: google/siglip2-base-patch16-224
    revision: null
    enabled: true

  dinov3_vit_b16:
    checkpoint: facebook/dinov3-vitb16-pretrain-lvd1689m
    revision: null
    enabled: true
    access_note: user_must_confirm_license_and_auth
'@
New-FileIfMissing -Path (Join-Path $ProjectRoot 'configs\models.yaml') -Content $ModelConfig

New-FileIfMissing -Path (Join-Path $ProjectRoot 'data\manifests\.gitkeep') -Content ''
New-FileIfMissing -Path (Join-Path $ProjectRoot 'artifacts\.gitkeep') -Content ''
New-FileIfMissing -Path (Join-Path $ProjectRoot 'artifacts\features\.gitkeep') -Content ''
New-FileIfMissing -Path (Join-Path $ProjectRoot 'artifacts\metrics\.gitkeep') -Content ''
New-FileIfMissing -Path (Join-Path $ProjectRoot 'artifacts\figures\.gitkeep') -Content ''
New-FileIfMissing -Path (Join-Path $ProjectRoot 'reports\runs\.gitkeep') -Content ''

New-FileIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\__init__.py') -Content '"""Vision encoder alignment research package."""'
New-FileIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\extractors\__init__.py') -Content '"""Vision encoder feature extractors."""'
New-FileIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\metrics\__init__.py') -Content '"""Representation alignment metrics."""'
New-FileIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\utils\__init__.py') -Content '"""Shared utilities."""'

New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\extractors\base.py') -Purpose 'Common extractor protocol and canonical feature contract.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\extractors\clip.py') -Purpose 'CLIP ViT feature extractor.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\extractors\siglip2.py') -Purpose 'SigLIP2 vision feature extractor.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\extractors\dinov3.py') -Purpose 'DINOv3 vision feature extractor.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\metrics\cka.py') -Purpose 'Memory-safe linear CKA implementation.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\metrics\procrustes.py') -Purpose 'Orthogonal Procrustes alignment and held-out evaluation.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\metrics\linear_alignment.py') -Purpose 'Ridge linear mapping between encoder representations.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'src\vision_encoder_alignment\metrics\patch_correspondence.py') -Purpose 'Patch-level spatial correspondence metrics.'

New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'scripts\extract_features.py') -Purpose 'CLI entry point for cached encoder feature extraction.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'scripts\compute_layer_similarity.py') -Purpose 'CLI entry point for layer-wise similarity matrices.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'scripts\fit_pairwise_mapping.py') -Purpose 'CLI entry point for pairwise Procrustes and ridge mappings.'
New-PythonStubIfMissing -Path (Join-Path $ProjectRoot 'scripts\evaluate_patch_alignment.py') -Purpose 'CLI entry point for patch-level alignment evaluation.'

Write-Host "[5/7] Initialize local Git repository" -ForegroundColor Cyan
$GitState = 'SKIPPED'
if (-not $SkipGitInit) {
    $GitCommand = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $GitCommand) {
        $Warnings.Add('Git was not found in PATH. Repository files were created, but git init was skipped.')
        $GitState = 'GIT_NOT_FOUND'
    }
    else {
        $ExistingTopLevel = $null
        try {
            $ExistingTopLevel = (& git -C $ProjectRoot rev-parse --show-toplevel 2>$null).Trim()
        }
        catch {
            $ExistingTopLevel = $null
        }

        if ($ExistingTopLevel) {
            $ResolvedProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
            $ResolvedTopLevel = [System.IO.Path]::GetFullPath($ExistingTopLevel).TrimEnd('\')

            if ($ResolvedTopLevel -ieq $ResolvedProjectRoot) {
                $GitState = 'ALREADY_INITIALIZED'
            }
            else {
                $Warnings.Add("Project path is already inside another Git repository: $ExistingTopLevel. Nested git init was not attempted.")
                $GitState = 'INSIDE_PARENT_REPOSITORY'
            }
        }
        else {
            & git -C $ProjectRoot init -b main | Out-Null
            if ($LASTEXITCODE -ne 0) {
                & git -C $ProjectRoot init | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    throw 'git init failed.'
                }
                & git -C $ProjectRoot branch -M main | Out-Null
            }
            $GitState = 'INITIALIZED'
        }
    }
}

Write-Host "[6/7] Write bootstrap report" -ForegroundColor Cyan
$ReportPath = Join-Path $ProjectRoot 'reports\runs\VEA-P0-T000-repository-bootstrap-script.md'
$Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'
$Branch = ''
$StatusText = ''

if (Get-Command git -ErrorAction SilentlyContinue) {
    try {
        $Branch = (& git -C $ProjectRoot branch --show-current 2>$null).Trim()
        $StatusText = (& git -C $ProjectRoot status --short 2>$null) -join "`n"
    }
    catch {
        $Branch = ''
        $StatusText = ''
    }
}

$WarningText = if ($Warnings.Count -gt 0) { ($Warnings | ForEach-Object { "- $_" }) -join "`n" } else { '- None' }
$ReportContent = @"
# VEA-P0-T000 — Repository Bootstrap Script Report

- Timestamp: $Timestamp
- Project root: $ProjectRoot
- Git state: $GitState
- Current branch: $Branch
- Created item count: $($CreatedItems.Count)
- Existing item count: $($SkippedItems.Count)

## Warnings

$WarningText

## Git status

    $($StatusText -replace "`n", "`n    ")

## Notes

- Existing files were not overwritten.
- No package installation, model download, dataset download, commit, remote creation, or push was performed.
- Python files are explicit non-functional stubs for later Codex tasks.
"@
Set-Content -LiteralPath $ReportPath -Value $ReportContent -Encoding utf8

Write-Host "[7/7] Complete" -ForegroundColor Green
Write-Host "Project root : $ProjectRoot"
Write-Host "Git state    : $GitState"
Write-Host "Created      : $($CreatedItems.Count) items"
Write-Host "Existing     : $($SkippedItems.Count) items"
Write-Host "Report       : $ReportPath"

if ($Warnings.Count -gt 0) {
    Write-Host "Warnings:" -ForegroundColor Yellow
    $Warnings | ForEach-Object { Write-Host "- $_" -ForegroundColor Yellow }
}

Write-Host ''
Write-Host 'Next command:' -ForegroundColor Cyan
Write-Host "Set-Location -LiteralPath '$ProjectRoot'"
Write-Host 'git status --short'
