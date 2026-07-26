# create-baseline.ps1 — 编译当前代码并将产物设为 sota-baseline
# 在每次优化任务开始前执行, 作为本次优化的对比基准

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$CurrentBin  = Join-Path $ProjectRoot "target\release\daiza-cli.exe"
$BaselineBin = Join-Path $ScriptDir "daiza-cli-baseline.exe"
$GoldenTokens = Join-Path $ScriptDir "golden_tokens.txt"

Write-Host "Building release binary..." -ForegroundColor Cyan
Push-Location $ProjectRoot
# cargo writes progress to stderr; redirect both to separate temp files
$tempOut = [System.IO.Path]::GetTempFileName()
$tempErr = [System.IO.Path]::GetTempFileName()
Start-Process -FilePath "cargo" -ArgumentList "build --release --locked" -NoNewWindow -Wait -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr
$buildOutput = (Get-Content $tempOut -Raw) + "`n" + (Get-Content $tempErr -Raw)
Remove-Item $tempOut, $tempErr -Force
Pop-Location

if ($buildOutput -match "error\b" -or -not (Test-Path $CurrentBin)) {
    Write-Host "error: build failed" -ForegroundColor Red
    Write-Host $buildOutput
    exit 1
}
Write-Host "  build OK" -ForegroundColor DarkGray

Copy-Item -Path $CurrentBin -Destination $BaselineBin -Force
Write-Host ""
Write-Host "Baseline binary: $BaselineBin" -ForegroundColor Green

# 生成 golden reference tokens (greedy decode, 用于后续正确性对比)
Write-Host "Generating golden reference tokens..." -ForegroundColor Cyan
$GgufPath = Join-Path $ProjectRoot "Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf"
$Prompt = "The capital of China is Beijing, and"

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $BaselineBin
$psi.Arguments = "--model `"$GgufPath`" --prompt `"$Prompt`" --max-tokens 64 --raw --greedy"
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
$psi.EnvironmentVariables["DAIZA_STREAM"] = "0"
$psi.EnvironmentVariables["DAIZA_DUMP_TOKENS"] = $GoldenTokens

$proc = [System.Diagnostics.Process]::Start($psi)
$stdout = $proc.StandardOutput.ReadToEnd()
$stderr = $proc.StandardError.ReadToEnd()
$proc.WaitForExit()

if (Test-Path $GoldenTokens) {
    Write-Host "Golden tokens: $GoldenTokens" -ForegroundColor Green
    Get-Content $GoldenTokens | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
} else {
    Write-Host "warning: golden tokens not generated" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Baseline ready. Use .\compare.ps1 to compare current vs baseline." -ForegroundColor Green
