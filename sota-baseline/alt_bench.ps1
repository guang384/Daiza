# alt_bench.ps1 — 交错对比 BL vs CR (消除热节流系统性偏差)
# 用法: .\alt_bench.ps1 -Dspark (默认 Basic 场景)
# 模式: BL1 -> CR1 -> BL2 -> CR2 -> ... (1 warmup + 5 measured = 12 runs)
param(
    [switch]$Dspark,
    [int]$Runs = 6,
    [int]$MaxTokens = 64,
    [int]$CooldownSec = 5
)
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$BaselineBin = Join-Path $ScriptDir "daiza-cli-baseline.exe"
$CurrentBin  = Join-Path $ProjectRoot "target\release\daiza-cli.exe"
$GgufPath = Join-Path $ProjectRoot "Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf"
$DsparkPath = Join-Path $ProjectRoot "Bonsai-27B-gguf\Bonsai-27B-dspark-Q4_1.gguf"

$scenario = if ($Dspark) { "DSpark" } else { "Basic" }
Write-Host ""
Write-Host "==============================================================" -ForegroundColor White
Write-Host "  Alternating benchmark: $scenario (BL vs CR)" -ForegroundColor White
Write-Host "==============================================================" -ForegroundColor White
Write-Host "  Runs: $Runs (1 warmup + $($Runs-1) measured), alternating BL/CR"
Write-Host ""

function Run-One {
    param([string]$Bin, [string]$Label)
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Bin
    if ($Dspark) {
        $psi.Arguments = "--model `"$GgufPath`" --dspark `"$DsparkPath`" --prompt `"The capital of China is Beijing, and`" --max-tokens $MaxTokens --raw --greedy"
    } else {
        $psi.Arguments = "--model `"$GgufPath`" --prompt `"The capital of China is Beijing, and`" --max-tokens $MaxTokens --raw --greedy"
    }
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $psi.EnvironmentVariables["DAIZA_STREAM"] = "0"
    $proc = [System.Diagnostics.Process]::Start($psi)
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()

    $prefillVal = -1.0; $decodeVal = -1.0; $toksVal = -1.0; $acceptVal = -1.0
    if ($stderr -match 'prefill\((\d+)t\)=([\d.]+)ms\s*\(~([\d.]+)ms/tok') { $prefillVal = [double]$Matches[3] }
    # DSpark 模式输出 "dspark decode(...)" 而非 "decode(...)"
    if ($stderr -match '(?:dspark\s+)?decode\((\d+)t\)=([\d.]+)ms\s*\(~([\d.]+)ms/tok\s*~([\d.]+)\s*tok/s') { $decodeVal = [double]$Matches[3]; $toksVal = [double]$Matches[4] }
    if ($stderr -match 'accept\s*[:=]\s*([\d.]+)') { $acceptVal = [double]$Matches[1] }
    return @($prefillVal, $decodeVal, $toksVal, $acceptVal)
}

$bP = [System.Collections.Generic.List[double]]::new()
$bD = [System.Collections.Generic.List[double]]::new()
$cP = [System.Collections.Generic.List[double]]::new()
$cD = [System.Collections.Generic.List[double]]::new()
$Warmup = 1

for ($round = 1; $round -le $Runs; $round++) {
    $isWarmup = ($round -le $Warmup)
    $status = if ($isWarmup) { "warmup" } else { "measured" }

    Write-Host "  Round $round/$Runs $status" -ForegroundColor Gray
    $r = Run-One $BaselineBin "BL"
    Write-Host ("    BL  prefill={0}ms  decode={1}ms  ({2} tok/s)" -f $r[0], $r[1], $r[2]) -ForegroundColor DarkGray
    if (-not $isWarmup) { $bP.Add($r[0]); $bD.Add($r[1]) }
    Start-Sleep -Seconds $CooldownSec

    $r = Run-One $CurrentBin "CR"
    Write-Host ("    CR  prefill={0}ms  decode={1}ms  ({2} tok/s)" -f $r[0], $r[1], $r[2]) -ForegroundColor DarkGray
    if (-not $isWarmup) { $cP.Add($r[0]); $cD.Add($r[1]) }
    Start-Sleep -Seconds $CooldownSec
}

function Calc-Min { param([System.Collections.Generic.List[double]]$V); if ($V.Count -eq 0) { return 0 }; return [math]::Round(($V.ToArray() | Measure-Object -Minimum).Minimum, 1) }
function Calc-Avg { param([System.Collections.Generic.List[double]]$V); if ($V.Count -eq 0) { return 0 }; return [math]::Round(($V.ToArray() | Measure-Object -Average).Average, 1) }

$bPMin = Calc-Min $bP; $cPMin = Calc-Min $cP
$bDMin = Calc-Min $bD; $cDMin = Calc-Min $cD
$bPAvg = Calc-Avg $bP; $cPAvg = Calc-Avg $cP
$bDAvg = Calc-Avg $bD; $cDAvg = Calc-Avg $cD

$pPct = if ($bPMin -gt 0) { [math]::Round(($cPMin - $bPMin) / $bPMin * 100, 1) } else { 0 }
$dPct = if ($bDMin -gt 0) { [math]::Round(($cDMin - $bDMin) / $bDMin * 100, 1) } else { 0 }

Write-Host ""
Write-Host "  +----------------+----------+----------+----------+----------+--------+" -ForegroundColor White
Write-Host "  | Metric (min)   |  BL min  |  CR min  |  BL avg  |  CR avg  | Delta  |" -ForegroundColor White
Write-Host "  +----------------+----------+----------+----------+----------+--------+" -ForegroundColor White
Write-Host ("  | prefill ms/tok | {0,8} | {1,8} | {2,8} | {3,8} | {4,5}%  |" -f $bPMin, $cPMin, $bPAvg, $cPAvg, $pPct)
Write-Host ("  | decode  ms/tok | {0,8} | {1,8} | {2,8} | {3,8} | {4,5}%  |" -f $bDMin, $cDMin, $bDAvg, $cDAvg, $dPct)
Write-Host "  +----------------+----------+----------+----------+----------+--------+" -ForegroundColor White
Write-Host ""
Write-Host "  Raw BL prefill: $($bP -join '  ')"
Write-Host "  Raw CR prefill: $($cP -join '  ')"
Write-Host "  Raw BL decode:  $($bD -join '  ')"
Write-Host "  Raw CR decode:  $($cD -join '  ')"
Write-Host ""
Write-Host "[summary] scenario=$scenario bl_prefill=$bPMin cr_prefill=$cPMin prefill_pct=$pPct bl_decode=$bDMin cr_decode=$cDMin decode_pct=$dPct"
