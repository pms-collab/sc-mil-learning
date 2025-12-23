# scripts/run_all.ps1
# End-to-end runner for scMIL pipeline (PowerShell 5.1 compatible)
# download -> preprocess -> build_bags -> train -> eval -> check_leakage

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,

    [Parameter(Mandatory = $true)]
    [string]$Config,

    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# -------------------------
# Helpers
# -------------------------
function Assert-File {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$Message = ""
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        if ([string]::IsNullOrWhiteSpace($Message)) {
            throw "Missing required file: $Path"
        } else {
            throw "$Message`nMissing required file: $Path"
        }
    }
}

function Assert-Dir {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$Message = ""
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        if ([string]::IsNullOrWhiteSpace($Message)) {
            throw "Missing required directory: $Path"
        } else {
            throw "$Message`nMissing required directory: $Path"
        }
    }
}

function Ensure-Dir {
    param([Parameter(Mandatory = $true)][string]$Path)
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Get-FullPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$BaseDir
    )
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $Path))
}

function Format-CmdLine {
    param([Parameter(Mandatory = $true)][string[]]$Parts)

    $fmt = New-Object System.Collections.Generic.List[string]
    foreach ($p in $Parts) {
        if ($null -eq $p) { continue }
        $s = [string]$p
        if ($s -match '\s' -or $s -match '"') {
            $s = $s -replace '"', '\"'
            $fmt.Add(('"' + $s + '"'))
        } else {
            $fmt.Add($s)
        }
    }
    return ($fmt -join " ")
}

function Run-Step {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$LogPath,
        [Parameter(Mandatory = $true)][string[]]$CmdParts
    )

    Ensure-Dir (Split-Path -Parent $LogPath)

    if (Test-Path -LiteralPath $LogPath) {
        Remove-Item -LiteralPath $LogPath -Force
    }

    $cmdLine = Format-CmdLine -Parts $CmdParts

    # Always record command line first (diagnose empty-output problems)
    ("[cmd] " + $cmdLine) | Out-File -FilePath $LogPath -Encoding UTF8 -Force

    Write-Host ("==> " + $Name)
    Write-Host ("    " + $cmdLine)
    Write-Host ("    log: " + $LogPath)

    $exe = $CmdParts[0]
    $args = @()
    if ($CmdParts.Count -gt 1) {
        $args = $CmdParts[1..($CmdParts.Count - 1)]
    }

    $oldEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $exe @args 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $text = $_.ToString()
            } else {
                $text = [string]$_
            }
            $text = $text.TrimEnd("`r", "`n")
            $text | Out-File -FilePath $LogPath -Append -Encoding UTF8
            $text
        } 
    }
    finally {
        $ErrorActionPreference = $oldEap
    }


    $code = $LASTEXITCODE
    if ($code -ne 0) {
        throw "Step '$Name' failed (exit=$code). See log: $LogPath"
    }
}

function Get-DatasetInfo {
    param(
        [Parameter(Mandatory = $true)][string]$PythonExePath,
        [Parameter(Mandatory = $true)][string]$ConfigPath
    )

    $tmpDir = [System.IO.Path]::GetTempPath()
    $tmp = Join-Path $tmpDir ("scmil_parse_cfg_" + [System.Guid]::NewGuid().ToString("N") + ".py")

    $code = @'
import json, sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg = cfg or {}
data = (cfg.get("data", {}) or {})
root = data.get("root", "data")
dataset = str(data.get("dataset", "") or "")

print(json.dumps({"root": root, "dataset": dataset}))
'@

    $code | Out-File -FilePath $tmp -Encoding UTF8 -Force

    try {
        # capture stdout+stderr (so failure shows diagnostics)
        $out = & $PythonExePath $tmp $ConfigPath 2>&1
        $exit = $LASTEXITCODE
        if ($exit -ne 0) {
            throw ("Failed to parse config via python/yaml. Check PythonExe + environment.`n" + ($out -join "`n"))
        }

        # take last non-empty line (defensive)
        $line = ($out | ForEach-Object { $_.ToString() } | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -Last 1)
        if ([string]::IsNullOrWhiteSpace($line)) {
            throw "Config parse returned empty output. Unexpected."
        }
        return ($line | ConvertFrom-Json)
    }
    finally {
        if (Test-Path -LiteralPath $tmp) {
            Remove-Item -LiteralPath $tmp -Force
        }
    }
}

# -------------------------
# Resolve repo root first (so relative -Config/-RunDir are anchored correctly)
# -------------------------
$ScriptDir = $PSScriptRoot
$RepoRoot  = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path

# Resolve input paths relative to repo root
$PythonExeAbs = Get-FullPath -Path $PythonExe -BaseDir $RepoRoot
$ConfigAbs    = Get-FullPath -Path $Config    -BaseDir $RepoRoot
$RunDirAbs    = Get-FullPath -Path $RunDir    -BaseDir $RepoRoot

if (-not (Test-Path -LiteralPath $PythonExeAbs -PathType Leaf)) {
    throw "PythonExe not found: $PythonExeAbs"
}
if (-not (Test-Path -LiteralPath $ConfigAbs -PathType Leaf)) {
    throw "Config not found: $ConfigAbs"
}

# Ensure run/log dirs
Ensure-Dir $RunDirAbs
$LogDir = Join-Path $RunDirAbs "logs"
Ensure-Dir $LogDir

# Move into repo root so "-m src...." works reliably
Push-Location $RepoRoot
try {
    # Parse dataset/root from YAML (avoid hardcoding)
    $info = Get-DatasetInfo -PythonExePath $PythonExeAbs -ConfigPath $ConfigAbs
    $DataRoot = [string]$info.root
    $Dataset  = [string]$info.dataset

    if ([string]::IsNullOrWhiteSpace($Dataset)) {
        throw "Config parse produced empty data.dataset. Fix configs/base.yaml."
    }
    if ([string]::IsNullOrWhiteSpace($DataRoot)) {
        $DataRoot = "data"
    }

    # DataRoot can be relative; anchor to repo root for artifact assertions
    $DataRootAbs = Get-FullPath -Path $DataRoot -BaseDir $RepoRoot

    # Expected artifacts (derived)
    $RawDir      = Join-Path $DataRootAbs (Join-Path "raw" $Dataset)
    $RawH5ad     = Join-Path $RawDir "raw.h5ad"
    $RawMarker   = $RawH5ad + ".ok"

    $PreprocessOut = Join-Path $RunDirAbs "preprocess"
    $ProcessedH5ad = Join-Path $PreprocessOut (Join-Path "artifacts" "processed.h5ad")

    $BagsOut       = Join-Path $RunDirAbs "bags"
    $BagsNpz       = Join-Path $BagsOut "bags.npz"
    $BagsMeta      = Join-Path $BagsOut "bags_meta.csv"
    $SplitBags     = Join-Path $BagsOut "split_bags.csv"
    $BagsMarker    = Join-Path $BagsOut "bags.ok"
    $BagsTableGz   = Join-Path $BagsOut (Join-Path "artifacts" "bags_table.csv.gz")

    $TrainOut      = Join-Path $RunDirAbs (Join-Path "train" "baseline")
    $BestCkpt      = Join-Path $TrainOut (Join-Path "checkpoints" "best.pt")
    $TrainSummary  = Join-Path $TrainOut "summary.json"
    $TrainHistory  = Join-Path $TrainOut "history.csv"

    $EvalOut       = Join-Path $RunDirAbs (Join-Path "eval" "test")
    $EvalPred      = Join-Path $EvalOut "predictions.csv"
    $EvalMetrics   = Join-Path $EvalOut "metrics.json"

    $LeakageOutDir = Join-Path $RunDirAbs "leakage"
    $LeakageReport = Join-Path $LeakageOutDir "report.json"

    # Force behavior: wipe run outputs (download is controlled by its own --force)
    if ($Force) {
        Write-Host "==> FORCE enabled: removing run outputs under RunDir (preprocess/bags/train/eval/leakage)."
        foreach ($p in @(
            $PreprocessOut,
            $BagsOut,
            (Join-Path $RunDirAbs "train"),
            (Join-Path $RunDirAbs "eval"),
            $LeakageOutDir
        )) {
            if (Test-Path -LiteralPath $p) {
                Remove-Item -LiteralPath $p -Recurse -Force
            }
        }
        Ensure-Dir $PreprocessOut
        Ensure-Dir $BagsOut
        Ensure-Dir $TrainOut
        Ensure-Dir $EvalOut
        Ensure-Dir $LeakageOutDir
    }

    # -------------------------
    # Step 01: download (raw.h5ad)
    # -------------------------
    $cmdDownload = @()
    $cmdDownload += $PythonExeAbs
    $cmdDownload += "-u"
    $cmdDownload += "-m"
    $cmdDownload += "src.data.download"
    $cmdDownload += "--config"
    $cmdDownload += $ConfigAbs
    if ($Force) { $cmdDownload += "--force" }

    Run-Step -Name "01_download" -LogPath (Join-Path $LogDir "01_download.log") -CmdParts $cmdDownload

    Assert-File -Path $RawH5ad   -Message "download did not produce raw.h5ad as expected."
    Assert-File -Path $RawMarker -Message "download did not produce raw marker (.ok) as expected."

    # -------------------------
    # Step 02: preprocess (processed.h5ad)
    # -------------------------
    $cmdPreprocess = @()
    $cmdPreprocess += $PythonExeAbs
    $cmdPreprocess += "-u"
    $cmdPreprocess += "-m"
    $cmdPreprocess += "src.preprocess"
    $cmdPreprocess += "--config"
    $cmdPreprocess += $ConfigAbs
    $cmdPreprocess += "--out"
    $cmdPreprocess += $PreprocessOut

    Run-Step -Name "02_preprocess" -LogPath (Join-Path $LogDir "02_preprocess.log") -CmdParts $cmdPreprocess

    Assert-File -Path $ProcessedH5ad -Message "preprocess did not produce artifacts/processed.h5ad."

    # -------------------------
    # Step 03: build_bags
    # -------------------------
    $cmdBuildBags = @()
    $cmdBuildBags += $PythonExeAbs
    $cmdBuildBags += "-u"
    $cmdBuildBags += "-m"
    $cmdBuildBags += "src.build_bags"
    $cmdBuildBags += "--config"
    $cmdBuildBags += $ConfigAbs
    $cmdBuildBags += "--out"
    $cmdBuildBags += $BagsOut
    $cmdBuildBags += "--preprocess_out"
    $cmdBuildBags += $PreprocessOut
    if ($Force) { $cmdBuildBags += "--force" }

    Run-Step -Name "03_build_bags" -LogPath (Join-Path $LogDir "03_build_bags.log") -CmdParts $cmdBuildBags

    Assert-File -Path $BagsNpz     -Message "build_bags did not produce bags.npz."
    Assert-File -Path $BagsMeta    -Message "build_bags did not produce bags_meta.csv."
    Assert-File -Path $SplitBags   -Message "build_bags did not produce split_bags.csv."
    Assert-File -Path $BagsMarker  -Message "build_bags did not produce bags.ok."
    Assert-File -Path $BagsTableGz -Message "build_bags did not produce artifacts/bags_table.csv.gz."

    # -------------------------
    # Step 04: train
    # -------------------------
    $cmdTrain = @()
    $cmdTrain += $PythonExeAbs
    $cmdTrain += "-u"
    $cmdTrain += "-m"
    $cmdTrain += "src.train"
    $cmdTrain += "--config"
    $cmdTrain += $ConfigAbs
    $cmdTrain += "--bags_dir"
    $cmdTrain += $BagsOut
    $cmdTrain += "--out"
    $cmdTrain += $TrainOut
    if ($Force) { $cmdTrain += "--force" }

    Run-Step -Name "04_train" -LogPath (Join-Path $LogDir "04_train.log") -CmdParts $cmdTrain

    Assert-File -Path $BestCkpt     -Message "train did not produce checkpoints/best.pt."
    Assert-File -Path $TrainSummary -Message "train did not produce summary.json."
    Assert-File -Path $TrainHistory -Message "train did not produce history.csv."

    # -------------------------
    # Step 05: eval (default: test split only)
    # -------------------------
    $cmdEval = @()
    $cmdEval += $PythonExeAbs
    $cmdEval += "-u"
    $cmdEval += "-m"
    $cmdEval += "src.eval"
    $cmdEval += "--bags_dir"
    $cmdEval += $BagsOut
    $cmdEval += "--ckpt"
    $cmdEval += $BestCkpt
    $cmdEval += "--out"
    $cmdEval += $EvalOut
    $cmdEval += "--splits"
    $cmdEval += "test"

    Run-Step -Name "05_eval" -LogPath (Join-Path $LogDir "05_eval.log") -CmdParts $cmdEval

    Assert-File -Path $EvalPred    -Message "eval did not produce predictions.csv."
    Assert-File -Path $EvalMetrics -Message "eval did not produce metrics.json."

    # -------------------------
    # Step 06: check_leakage (writes report.json)
    # -------------------------
    Ensure-Dir $LeakageOutDir
    $cmdLeak = @()
    $cmdLeak += $PythonExeAbs
    $cmdLeak += "-u"
    $cmdLeak += "-m"
    $cmdLeak += "src.check_leakage"
    $cmdLeak += "--bags_dir"
    $cmdLeak += $BagsOut
    $cmdLeak += "--out"
    $cmdLeak += $LeakageReport

    Run-Step -Name "06_check_leakage" -LogPath (Join-Path $LogDir "06_check_leakage.log") -CmdParts $cmdLeak

    Assert-File -Path $LeakageReport -Message "check_leakage did not produce report.json."

    Write-Host ""
    Write-Host "DONE"
    Write-Host ("RunDir: " + $RunDirAbs)
    Write-Host ("Logs : " + $LogDir)
    Write-Host ("Raw  : " + $RawH5ad)
    Write-Host ("Proc : " + $ProcessedH5ad)
    Write-Host ("Bags : " + $BagsOut)
    Write-Host ("Ckpt : " + $BestCkpt)
    Write-Host ("Eval : " + $EvalOut)
    Write-Host ("Leak : " + $LeakageReport)
}
finally {
    Pop-Location
}
