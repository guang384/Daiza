# bench_inline.ps1 — 5 轮交错 BL vs CR benchmark
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$BaselineBin = Join-Path $ScriptDir "daiza-cli-baseline.exe"
$CurrentBin  = Join-Path $ProjectRoot "target\release\daiza-cli.exe"
$GgufPath    = Join-Path $ProjectRoot "..\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf"
$Prompt = "The capital of China is Beijing, and"
$MaxTokens = 64

function Run-Bench($Bin) {
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Bin
    $psi.Arguments = "`"$GgufPath`" `"$Prompt`" $MaxTokens --raw --greedy"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $psi.EnvironmentVariables["DAIZA_STREAM"] = "0"
    $proc = [System.Diagnostics.Process]::Start($psi)
    $stderr = $proc.StandardError.ReadToEnd()
    $proc.WaitForExit()
    $prefill = -1.0; $decode = -1.0
    if ($stderr -match 'prefill\((\d+)t\)=([\d.]+)ms\s*\(~([\d.]+)ms/tok') { $prefill = [double]$Matches[3] }
    if ($stderr -match 'decode\((\d+)t\)=([\d.]+)ms\s*\(~([\d.]+)ms/tok') { $decode = [double]$Matches[3] }
    return @($prefill, $decode)
}

$blP = @(); $blD = @(); $crP = @(); $crD = @()
$rounds = 6
for ($r = 1; $r -le $rounds; $r++) {
    $isWarmup = ($r -eq 1)
    $r1 = Run-Bench $BaselineBin
    Start-Sleep -Seconds 3
    $r2 = Run-Bench $CurrentBin
    Start-Sleep -Seconds 3
    if (-not $isWarmup) {
        $blP += $r1[0]; $blD += $r1[1]
        $crP += $r2[0]; $crD += $r2[1]
    }
    Write-Host ("Round {0}: BL P={1} D={2}  CR P={3} D={4}" -f $r, $r1[0], $r1[1], $r2[0], $r2[1])
}

$blPMin = ($blP | Measure-Object -Minimum).Minimum
$blDMin = ($blD | Measure-Object -Minimum).Minimum
$crPMin = ($crP | Measure-Object -Minimum).Minimum
$crDMin = ($crD | Measure-Object -Minimum).Minimum
$pPct = [math]::Round(($crPMin - $blPMin) / $blPMin * 100, 1)
$dPct = [math]::Round(($crDMin - $blDMin) / $blDMin * 100, 1)
Write-Host ""
Write-Host ("RESULT prefill: BL={0}  CR={1}  ({2}%)" -f $blPMin, $crPMin, $pPct)
Write-Host ("RESULT decode:  BL={0}  CR={1}  ({2}%)" -f $blDMin, $crDMin, $dPct)
Write-Host ""
Write-Host ("BL P raw: {0}" -f ($blP -join '  '))
Write-Host ("CR P raw: {0}" -f ($crP -join '  '))
Write-Host ("BL D raw: {0}" -f ($blD -join '  '))
Write-Host ("CR D raw: {0}" -f ($crD -join '  '))
