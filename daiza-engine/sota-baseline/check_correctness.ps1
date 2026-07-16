# check_correctness.ps1 — verify token IDs match golden reference
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$CurrentBin  = Join-Path $ProjectRoot "target\release\daiza-cli.exe"
$GoldenTokens = Join-Path $ScriptDir "golden_tokens.txt"
$CurrentTokens = Join-Path $ScriptDir "current_tokens.txt"
$GgufPath = Join-Path $ProjectRoot "..\Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf"
$Prompt = "The capital of China is Beijing, and"

$psi = [System.Diagnostics.ProcessStartInfo]::new()
$psi.FileName = $CurrentBin
$psi.Arguments = "`"$GgufPath`" `"$Prompt`" 64 --raw --greedy"
$psi.UseShellExecute = $false
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
$psi.EnvironmentVariables["DAIZA_STREAM"] = "0"
$psi.EnvironmentVariables["DAIZA_DUMP_TOKENS"] = $CurrentTokens

$proc = [System.Diagnostics.Process]::Start($psi)
$proc.StandardOutput.ReadToEnd() | Out-Null
$proc.WaitForExit()

if ((Test-Path $GoldenTokens) -and (Test-Path $CurrentTokens)) {
    $golden = Get-Content $GoldenTokens -Raw
    $current = Get-Content $CurrentTokens -Raw
    if ($golden -eq $current) {
        Write-Host "[PASS] Token IDs match golden reference" -ForegroundColor Green
    } else {
        Write-Host "[FAIL] Token IDs differ from golden reference!" -ForegroundColor Red
        Write-Host "--- golden ---" -ForegroundColor DarkGray
        Write-Host $golden -ForegroundColor DarkGray
        Write-Host "--- current ---" -ForegroundColor DarkGray
        Write-Host $current -ForegroundColor DarkGray
    }
} else {
    Write-Host "[SKIP] golden_tokens.txt not found" -ForegroundColor Yellow
}
