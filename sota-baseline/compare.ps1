# compare.ps1 — 对比当前编译版本 vs sota-baseline 的 e2e 性能 + 正确性
#
# 用法:
#   .\compare.ps1                         # 默认 4 轮/组 (1 warmup + 3 measured), Alternating
#   .\compare.ps1 -Runs 6                 # 6 轮/组 (1 warmup + 5 measured)
#   .\compare.ps1 -CheckCorrectness       # 额外运行 1 次 greedy decode 对比 token IDs
#
# 执行模式: Alternating (BL1→CR1→BL2→CR2→...)
#   优势: CPU 热节流对两组对称——如果 CPU 持续升温, BL1/CR1 都凉, BL2/CR2 都热。
#         消除"BL 先跑 CPU 凉, CR 后跑 CPU 热"的系统性偏差。
#
# 统计量用 min: 内存/计算 bound 场景下噪声单向放大, min 最接近纯计算时间

param(
    [int]$Runs = 4,
    [switch]$CheckCorrectness,
    [string]$Prompt = "The capital of China is Beijing, and",
    [int]$MaxTokens = 64,
    [int]$CooldownSec = 5
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$BaselineBin = Join-Path $ScriptDir "daiza-cli-baseline.exe"
$CurrentBin  = Join-Path $ProjectRoot "target\release\daiza-cli.exe"
$GoldenTokens = Join-Path $ScriptDir "golden_tokens.txt"
$GgufPath = Join-Path $ProjectRoot "Bonsai-27B-gguf\Bonsai-27B-Q1_0.gguf"

if (-not (Test-Path $BaselineBin)) {
    Write-Host "error: baseline binary not found: $BaselineBin" -ForegroundColor Red
    Write-Host "  Run .\create-baseline.ps1 first" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $CurrentBin)) {
    Write-Host "Building current release binary..." -ForegroundColor Cyan
    Push-Location $ProjectRoot
    $tempOut = [System.IO.Path]::GetTempFileName()
    $tempErr = [System.IO.Path]::GetTempFileName()
    Start-Process -FilePath "cargo" -ArgumentList "build --release" -NoNewWindow -Wait -RedirectStandardOutput $tempOut -RedirectStandardError $tempErr
    $buildOutput = (Get-Content $tempOut -Raw) + "`n" + (Get-Content $tempErr -Raw)
    Remove-Item $tempOut, $tempErr -Force
    Pop-Location
    if ($buildOutput -match "error\b" -or -not (Test-Path $CurrentBin)) {
        Write-Host "error: build failed" -ForegroundColor Red
        Write-Host $buildOutput
        exit 1
    }
}

$Warmup = 1
$Measured = $Runs - $Warmup
if ($Measured -lt 2) {
    Write-Host "error: need at least $Warmup+2 = $($Warmup+2) runs" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==============================================================" -ForegroundColor White
Write-Host "  daiza-cli: Current vs sota-baseline" -ForegroundColor White
Write-Host "==============================================================" -ForegroundColor White
Write-Host "  Baseline: $BaselineBin" -ForegroundColor Cyan
Write-Host "  Current:  $CurrentBin" -ForegroundColor Cyan
Write-Host "  Prompt:   `"$Prompt`"  --max-tokens $MaxTokens --raw --greedy"
Write-Host "  Runs:     $Runs (1 warmup + $Measured measured), Alternating"
Write-Host "  Cooldown: ${CooldownSec}s"
Write-Host "  Stats:    min (抗 outlier)" -ForegroundColor DarkGray
Write-Host ""

function Run-Bench {
    param([string]$Bin, [string]$Label, [bool]$IsWarmup,
          [System.Collections.Generic.List[double]]$PrefillList,
          [System.Collections.Generic.List[double]]$DecodeList,
          [System.Collections.Generic.List[double]]$ToksList)

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
    $proc.WaitForExit()

    $prefillVal = -1.0; $decodeVal = -1.0; $toksVal = -1.0
    if ($stderr -match 'prefill\((\d+)t\)=([\d.]+)ms\s*\(~([\d.]+)ms/tok') {
        $prefillVal = [double]$Matches[3]
    }
    if ($stderr -match 'decode\((\d+)t\)=([\d.]+)ms\s*\(~([\d.]+)ms/tok\s*~([\d.]+)\s*tok/s') {
        $decodeVal = [double]$Matches[3]
        $toksVal = [double]$Matches[4]
    }

    $status = if ($IsWarmup) { "warmup" } else { "measured" }
    $color = if ($IsWarmup) { "DarkGray" } else { "Gray" }
    Write-Host "    $Label $status prefill=$prefillVal ms/tok  decode=$decodeVal ms/tok ($toksVal tok/s)" -ForegroundColor $color

    if (-not $IsWarmup) {
        if ($prefillVal -gt 0) { $PrefillList.Add($prefillVal) }
        if ($decodeVal -gt 0) { $DecodeList.Add($decodeVal) }
        if ($toksVal -gt 0) { $ToksList.Add($toksVal) }
    }
}

$bPrefill = [System.Collections.Generic.List[double]]::new()
$bDecode = [System.Collections.Generic.List[double]]::new()
$bToks = [System.Collections.Generic.List[double]]::new()
$cPrefill = [System.Collections.Generic.List[double]]::new()
$cDecode = [System.Collections.Generic.List[double]]::new()
$cToks = [System.Collections.Generic.List[double]]::new()

$totalRuns = $Runs * 2
$runIdx = 0
for ($round = 1; $round -le $Runs; $round++) {
    $isWarmup = ($round -le $Warmup)
    $runIdx++
    Write-Host "  [$runIdx/$totalRuns] BL round $round" -ForegroundColor White
    Run-Bench $BaselineBin "BL" $isWarmup $bPrefill $bDecode $bToks
    if ($round -lt $Runs -and $CooldownSec -gt 0) { Start-Sleep -Seconds $CooldownSec }

    $runIdx++
    Write-Host "  [$runIdx/$totalRuns] CR round $round" -ForegroundColor White
    Run-Bench $CurrentBin "CR" $isWarmup $cPrefill $cDecode $cToks
    if ($round -lt $Runs -and $CooldownSec -gt 0) { Start-Sleep -Seconds $CooldownSec }
}

function Calc-Min { param([System.Collections.Generic.List[double]]$V); if ($V.Count -eq 0) { return "N/A" }; return [math]::Round(($V.ToArray() | Measure-Object -Minimum).Minimum, 2) }
function Calc-Avg { param([System.Collections.Generic.List[double]]$V); if ($V.Count -eq 0) { return "N/A" }; return [math]::Round(($V.ToArray() | Measure-Object -Average).Average, 2) }
function Calc-Range { param([System.Collections.Generic.List[double]]$V); if ($V.Count -eq 0) { return "N/A" }; $s = $V.ToArray() | Measure-Object -Minimum -Maximum; return [math]::Round($s.Maximum - $s.Minimum, 2) }
function Calc-Pct { param([string]$O, [string]$N); if ($O -eq "N/A" -or $N -eq "N/A" -or $O -eq "0") { return "N/A" }; return [math]::Round(([double]$N - [double]$O) / [double]$O * 100, 1) }
function Get-Verdict {
    param([string]$P)
    if ($P -eq "N/A") { return "N/A" }
    $v = [double]$P; $a = [math]::Abs($v)
    if ($a -lt 5) { return "NOISE" }
    if ($v -lt 0) { return "FASTER" }
    return "SLOWER"
}

$bPMin = Calc-Min $bPrefill; $cPMin = Calc-Min $cPrefill
$bDMin = Calc-Min $bDecode;  $cDMin = Calc-Min $cDecode
$bTMin = Calc-Min $bToks;    $cTMin = Calc-Min $cToks
$bPAvg = Calc-Avg $bPrefill; $cPAvg = Calc-Avg $cPrefill
$bDAvg = Calc-Avg $bDecode;  $cDAvg = Calc-Avg $cDecode
$bPRange = Calc-Range $bPrefill; $cPRange = Calc-Range $cPrefill
$bDRange = Calc-Range $bDecode;  $cDRange = Calc-Range $cDecode

$prefillPct = Calc-Pct $bPMin $cPMin
$decodePct = Calc-Pct $bDMin $cDMin
$toksPct = Calc-Pct $bTMin $cTMin
$prefillVerdict = Get-Verdict $prefillPct
$decodeVerdict = Get-Verdict $decodePct

function Format-Pct { param([string]$P, [string]$V); switch ($V) { "FASTER" { Write-Host -NoNewline -ForegroundColor Green "$P%" } "SLOWER" { Write-Host -NoNewline -ForegroundColor Red "$P%" } "NOISE" { Write-Host -NoNewline -ForegroundColor Yellow "$P% (noise)" } default { Write-Host -NoNewline "$P%" } } }
function Format-Verdict { param([string]$V); switch ($V) { "FASTER" { Write-Host -NoNewline -ForegroundColor Green "真优化" } "SLOWER" { Write-Host -NoNewline -ForegroundColor Red "回归" } "NOISE" { Write-Host -NoNewline -ForegroundColor Yellow "噪声区间" } default { Write-Host -NoNewline "$V" } } }

Write-Host ""
Write-Host "+============================================================+" -ForegroundColor White
Write-Host "|                      BENCHMARK RESULT                     |" -ForegroundColor White
Write-Host "+============================================================+" -ForegroundColor White
Write-Host ""
Write-Host "  +----------------+-------------+-------------+----------+" -ForegroundColor White
Write-Host "  | Metric (min)   |  Baseline   |   Current   |  Delta   |" -ForegroundColor White
Write-Host "  +----------------+-------------+-------------+----------+" -ForegroundColor White

Write-Host -NoNewline "  | prefill ms/tok |"
Write-Host -NoNewline ("{0,10}ms |" -f $bPMin)
Write-Host -NoNewline ("{0,10}ms | " -f $cPMin)
Format-Pct $prefillPct $prefillVerdict
Write-Host "    |"

Write-Host -NoNewline "  | decode ms/tok  |"
Write-Host -NoNewline ("{0,10}ms |" -f $bDMin)
Write-Host -NoNewline ("{0,10}ms | " -f $cDMin)
Format-Pct $decodePct $decodeVerdict
Write-Host "    |"

$bTps = if ($bTMin -ne "N/A") { [math]::Round([double]$bTMin, 2) } else { "N/A" }
$cTps = if ($cTMin -ne "N/A") { [math]::Round([double]$cTMin, 2) } else { "N/A" }
Write-Host -NoNewline "  | decode tok/s   |"
Write-Host -NoNewline ("{0,10}  |" -f $bTps)
Write-Host -NoNewline ("{0,10}  | " -f $cTps)
Format-Pct $toksPct $decodeVerdict
Write-Host "    |"

Write-Host "  +----------------+-------------+-------------+----------+" -ForegroundColor White
Write-Host ""

# Jitter
Write-Host "  Jitter diagnostics (range = max - min)" -ForegroundColor White
Write-Host "    prefill: BL range=$bPRange  CR range=$cPRange"
Write-Host "    decode:  BL range=$bDRange  CR range=$cDRange"
Write-Host "    raw BL prefill: $($bPrefill -join '  ')"
Write-Host "    raw CR prefill: $($cPrefill -join '  ')"
Write-Host "    raw BL decode:  $($bDecode -join '  ')"
Write-Host "    raw CR decode:  $($cDecode -join '  ')"
Write-Host ""

# 正确性检查
if ($CheckCorrectness) {
    Write-Host "  Correctness check (greedy decode token IDs)" -ForegroundColor White
    $currentTokens = Join-Path $ScriptDir "current_tokens.txt"
    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $CurrentBin
    $psi.Arguments = "--model `"$GgufPath`" --prompt `"$Prompt`" --max-tokens $MaxTokens --raw --greedy"
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $psi.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $psi.EnvironmentVariables["DAIZA_STREAM"] = "0"
    $psi.EnvironmentVariables["DAIZA_DUMP_TOKENS"] = $currentTokens
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.StandardOutput.ReadToEnd() | Out-Null
    $proc.WaitForExit()

    if (Test-Path $GoldenTokens -and (Test-Path $currentTokens)) {
        $golden = Get-Content $GoldenTokens -Raw
        $current = Get-Content $currentTokens -Raw
        if ($golden -eq $current) {
            Write-Host "    [PASS] Token IDs match golden reference" -ForegroundColor Green
        } else {
            Write-Host "    [FAIL] Token IDs differ from golden reference!" -ForegroundColor Red
            Write-Host "    --- golden ---" -ForegroundColor DarkGray
            Write-Host "    $golden" -ForegroundColor DarkGray
            Write-Host "    --- current ---" -ForegroundColor DarkGray
            Write-Host "    $current" -ForegroundColor DarkGray
        }
    } else {
        Write-Host "    [SKIP] golden_tokens.txt not found, run create-baseline.ps1 first" -ForegroundColor Yellow
    }
    Write-Host ""
}

# 判定
Write-Host "  Verdict" -ForegroundColor White
Write-Host "  ----------------------------------------"
$allOk = $true
foreach ($entry in @(@($prefillVerdict, $prefillPct, "prefill"), @($decodeVerdict, $decodePct, "decode"))) {
    $v = $entry[0]; $p = $entry[1]; $name = $entry[2]
    Write-Host -NoNewline "  $name "
    Format-Pct $p $v
    switch ($v) {
        "FASTER" { Write-Host " -> commit OK" -ForegroundColor Green }
        "NOISE"  { Write-Host " -> ask user" -ForegroundColor Yellow }
        "SLOWER" { Write-Host " -> MUST revert" -ForegroundColor Red; $allOk = $false }
    }
}
Write-Host ""

# 机器可解析摘要
Write-Host "[summary] bl_prefill=$bPMin cr_prefill=$cPMin prefill_pct=$prefillPct bl_decode=$bDMin cr_decode=$cDMin decode_pct=$decodePct bl_tps=$bTMin cr_tps=$cTMin"
