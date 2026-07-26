# bench.ps1 — 单二进制 e2e benchmark, 多轮取 min (抗热节流噪声)
#
# 用法:
#   .\bench.ps1                          # 默认 4 轮 (1 warmup + 3 measured)
#   .\bench.ps1 -Runs 6                  # 6 轮 (1 warmup + 5 measured)
#   .\bench.ps1 -Bin path\to\daiza-cli.exe
#   .\bench.ps1 -Prompt "Hello" -MaxTokens 128
#
# 输出: prefill/decode 的 min/avg/max ms/tok + tok/s
#
# 统计量用 min: 内存/计算 bound 场景下噪声单向放大 (热节流只让结果变慢),
# min 最接近纯计算时间, 抗 outlier 能力远强于 mean.

param(
    [int]$Runs = 4,
    [string]$Bin = "",
    [string]$Prompt = "The capital of China is Beijing, and",
    [int]$MaxTokens = 64,
    [int]$CooldownSec = 5
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

if ($Bin -eq "") {
    $Bin = Join-Path $ProjectRoot "target\release\daiza-cli.exe"
}

if (-not (Test-Path $Bin)) {
    Write-Host "Building release binary..." -ForegroundColor Cyan
    Push-Location $ProjectRoot
    $tempOut = [System.IO.Path]::GetTempFileName()
    $tempErr = [System.IO.Path]::GetTempFileName()
    Start-Process -FilePath "cargo" -ArgumentList "build --release --locked" -NoNewWindow -Wait -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr
    $buildOutput = (Get-Content $tempOut -Raw) + "`n" + (Get-Content $tempErr -Raw)
    Remove-Item $tempOut, $tempErr -Force
    Pop-Location
    if ($buildOutput -match "error\b" -or -not (Test-Path $Bin)) {
        Write-Host "error: build failed" -ForegroundColor Red
        Write-Host $buildOutput
        exit 1
    }
}

$GgufPath = Join-Path $ProjectRoot "Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf"
if (-not (Test-Path $GgufPath)) {
    Write-Host "error: GGUF model not found at $GgufPath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor White
Write-Host "  daiza-cli e2e benchmark" -ForegroundColor White
Write-Host "==============================================================" -ForegroundColor White
Write-Host "  Binary:    $Bin"
Write-Host "  Model:     $GgufPath"
Write-Host "  Prompt:    `"$Prompt`"  --max-tokens $MaxTokens"
Write-Host "  Runs:      $Runs (1 warmup + $($Runs - 1) measured)"
Write-Host "  Cooldown:  ${CooldownSec}s"
Write-Host "  Stats:     min (IO-bound 噪声单向放大, min 抗 outlier)"
Write-Host ""

$Warmup = 1
$Measured = $Runs - $Warmup
if ($Measured -lt 1) {
    Write-Host "error: need at least 2 runs (1 warmup + 1 measured)" -ForegroundColor Red
    exit 1
}

$prefillMs = [System.Collections.Generic.List[double]]::new()
$decodeMs = [System.Collections.Generic.List[double]]::new()
$decodeToks = [System.Collections.Generic.List[double]]::new()

for ($round = 1; $round -le $Runs; $round++) {
    $isWarmup = ($round -le $Warmup)
    $status = if ($isWarmup) { "warmup" } else { "measuring" }
    Write-Host "  [$round/$Runs] $status..." -ForegroundColor $(if ($isWarmup) { "DarkGray" } else { "Gray" })

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Bin
    $psi.Arguments = "--model `"$GgufPath`" --prompt `"$Prompt`" --max-tokens $MaxTokens --raw --greedy"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $psi.EnvironmentVariables["DAIZA_STREAM"] = "0"

    $proc = [System.Diagnostics.Process]::Start($psi)
    $stderr = $proc.StandardError.ReadToEnd()
    $stdout = $proc.StandardOutput.ReadToEnd()
    $proc.WaitForExit()

    # Parse: [bench] load=...ms prefill(Nt)=...ms (~Xms/tok)
    #        [bench] decode(Nt)=...ms (~Xms/tok ~Y tok/s)
    $prefillVal = -1.0
    $decodeVal = -1.0
    $decodeToksVal = -1.0

    if ($stderr -match 'prefill\((\d+)t\)=([\d.]+)ms\s*\(~([\d.]+)ms/tok') {
        $prefillVal = [double]$Matches[3]
    }
    if ($stderr -match 'decode\((\d+)t\)=([\d.]+)ms\s*\(~([\d.]+)ms/tok\s*~([\d.]+)\s*tok/s') {
        $decodeVal = [double]$Matches[3]
        $decodeToksVal = [double]$Matches[4]
    }

    if (-not $isWarmup) {
        if ($prefillVal -gt 0) { $prefillMs.Add($prefillVal) }
        if ($decodeVal -gt 0) { $decodeMs.Add($decodeVal) }
        if ($decodeToksVal -gt 0) { $decodeToks.Add($decodeToksVal) }
        Write-Host "    prefill: $prefillVal ms/tok, decode: $decodeVal ms/tok ($decodeToksVal tok/s)"
    } else {
        Write-Host "    (warmup) prefill: $prefillVal ms/tok, decode: $decodeVal ms/tok"
    }

    if ($round -lt $Runs -and $CooldownSec -gt 0) {
        Start-Sleep -Seconds $CooldownSec
    }
}

function Calc-Stat {
    param([System.Collections.Generic.List[double]]$Values, [string]$Stat)
    if ($Values.Count -eq 0) { return "N/A" }
    $arr = $Values.ToArray()
    switch ($Stat) {
        "min"  { return [math]::Round(($arr | Measure-Object -Minimum).Minimum, 2) }
        "avg"  { return [math]::Round(($arr | Measure-Object -Average).Average, 2) }
        "max"  { return [math]::Round(($arr | Measure-Object -Maximum).Maximum, 2) }
        "range" {
            $s = $arr | Measure-Object -Minimum -Maximum
            return [math]::Round($s.Maximum - $s.Minimum, 2)
        }
    }
}

Write-Host ""
Write-Host "  +--------------+-------+-------+-------+-------+"
Write-Host "  | Metric       |  min  |  avg  |  max  | range |"
Write-Host "  +--------------+-------+-------+-------+-------+"
$prefillMin = Calc-Stat $prefillMs "min"
$prefillAvg = Calc-Stat $prefillMs "avg"
$prefillMax = Calc-Stat $prefillMs "max"
$prefillRange = Calc-Stat $prefillMs "range"
$decodeMin = Calc-Stat $decodeMs "min"
$decodeAvg = Calc-Stat $decodeMs "avg"
$decodeMax = Calc-Stat $decodeMs "max"
$decodeRange = Calc-Stat $decodeMs "range"
$toksMin = Calc-Stat $decodeToks "min"
$toksAvg = Calc-Stat $decodeToks "avg"

Write-Host ("  | {0,-12} | {1,5} | {2,5} | {3,5} | {4,5} |" -f "prefill ms/tok", $prefillMin, $prefillAvg, $prefillMax, $prefillRange)
Write-Host ("  | {0,-12} | {1,5} | {2,5} | {3,5} | {4,5} |" -f "decode ms/tok", $decodeMin, $decodeAvg, $decodeMax, $decodeRange)
Write-Host ("  | {0,-12} | {1,5} | {2,5} | {3,-5} | {4,-5} |" -f "decode tok/s", $toksMin, $toksAvg, "", "")
Write-Host "  +--------------+-------+-------+-------+-------+"
Write-Host ""

# 可靠性: range > 5% of min 标记 NOISY
function Get-Reliability {
    param([string]$Min, [string]$Range)
    if ($Min -eq "N/A" -or $Range -eq "N/A") { return "N/A" }
    $m = [double]$Min; $r = [double]$Range
    if ($m -le 0) { return "N/A" }
    if ($r / $m * 100 -gt 5) { return "NOISY" }
    return "OK"
}
$prefillRel = Get-Reliability $prefillMin $prefillRange
$decodeRel = Get-Reliability $decodeMin $decodeRange
Write-Host "  Reliability: prefill=$prefillRel, decode=$decodeRel (range<=5% of min = OK)"
Write-Host ""

# 原始数据
Write-Host "  Raw samples (warmup excluded)" -ForegroundColor DarkGray
Write-Host "    prefill ms/tok: $($prefillMs -join '  ')" -ForegroundColor DarkGray
Write-Host "    decode  ms/tok: $($decodeMs -join '  ')" -ForegroundColor DarkGray
Write-Host "    decode  tok/s:  $($decodeToks -join '  ')" -ForegroundColor DarkGray
Write-Host ""

# 输出机器可解析的摘要行 (供 compare.ps1 抓取)
Write-Host "[summary] prefill_min_ms_per_tok=$prefillMin decode_min_ms_per_tok=$decodeMin decode_min_tok_per_s=$toksMin"
