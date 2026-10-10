param(
  [Parameter(Mandatory = $true)][string]$Tag,
  [string]$BaseUrl = "http://127.0.0.1:8099",
  [string]$OutDir = "build-win/r52",
  [string]$CompareTo = ""
)
# Deterministic greedy golden probe (top_k=1 pins argmax). Five fixed prompts,
# 160 tokens each, hash of content||reasoning_content. Compare two tags with -CompareTo.
#
# Capture goldens on a FRESH server that has served no other request: cached
# prefix KV rows carry their producing prefill's reduction shape (ulp-level),
# so a warmup request with a different prompt shape shifts every later greedy
# trajectory. Goldens taken after unrelated requests do not reproduce.
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$prompts = @(
  (([string][char]0x7528) + [char]0x4E00 + [char]0x53E5 + [char]0x8BDD + [char]0x89E3 + [char]0x91CA + [char]0x4EC0 + [char]0x4E48 + [char]0x662F + [char]0x5F20 + [char]0x91CF + [char]0x5E76 + [char]0x884C + [char]0x3002),
  (([string][char]0x628A) + [char]0x201C + [char]0x4ECA + [char]0x5929 + [char]0x5929 + [char]0x6C14 + [char]0x5F88 + [char]0x597D + [char]0xFF0C + [char]0x6211 + [char]0x4EEC + [char]0x53BB + [char]0x516C + [char]0x56ED + [char]0x6563 + [char]0x6B65 + [char]0x3002 + [char]0x201D + [char]0x7FFB + [char]0x8BD1 + [char]0x6210 + [char]0x82F1 + [char]0x6587 + [char]0x3002),
  'Write a Python function that returns the n-th Fibonacci number iteratively.',
  (('3+4*5 ') + [char]0x7B49 + [char]0x4E8E + [char]0x591A + [char]0x5C11 + [char]0xFF1F + [char]0x53EA + [char]0x56DE + [char]0x7B54 + [char]0x6570 + [char]0x5B57 + [char]0x3002),
  'List three differences between TCP and UDP, one line each.'
)
$rows = @()
for ($i = 0; $i -lt $prompts.Count; ++$i) {
  $bodyObj = @{
    model      = "qwen3.8-27b"
    messages   = @(@{ role = "user"; content = $prompts[$i] })
    max_tokens = 160
    temperature = 0
    top_k      = 1
    top_p      = 1.0
  }
  $json = $bodyObj | ConvertTo-Json -Depth 6 -Compress
  $bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
  $resp = Invoke-RestMethod -Uri "$BaseUrl/v1/chat/completions" -Method Post -ContentType "application/json; charset=utf-8" -Body $bytes -TimeoutSec 300
  if (-not $resp.choices) {
    Write-Host ("req{0} ERROR {1}" -f $i, ($resp | ConvertTo-Json -Compress))
    $rows += [ordered]@{ i = $i; error = "no choices" }
    continue
  }
  $ch = $resp.choices[0]
  $m = $ch.message
  $text = [string]$m.content + "||" + [string]$m.reasoning_content
  $sha = [System.Security.Cryptography.SHA256]::Create()
  $hash = -join ($sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($text)) | ForEach-Object { $_.ToString("x2") })
  $hash = $hash.Substring(0, 16)
  Write-Host ("req{0} finish={1} hash={2} len={3}" -f $i, $ch.finish_reason, $hash, $text.Length)
  $rows += [ordered]@{ i = $i; finish = $ch.finish_reason; hash = $hash; len = $text.Length; text = $text }
}
$out = Join-Path $OutDir ("ab-" + $Tag + ".jsonl")
$lines = foreach ($r in $rows) { ($r | ConvertTo-Json -Compress -Depth 4) }
Set-Content -Path $out -Value $lines -Encoding utf8
Write-Host "wrote $out"
if ($CompareTo) {
  $ref = Get-Content (Join-Path $OutDir ("ab-" + $CompareTo + ".jsonl")) | ForEach-Object { $_ | ConvertFrom-Json }
  $ok = $true
  for ($i = 0; $i -lt $rows.Count; ++$i) {
    if ($rows[$i].hash -ne $ref[$i].hash) {
      $ok = $false
      Write-Host ("   DIFF req{0} ref {1}({2}) vs {3}({4})" -f $i, $ref[$i].hash, $ref[$i].len, $rows[$i].hash, $rows[$i].len)
    }
  }
  Write-Host ("{0} vs {1}: {2}" -f $Tag, $CompareTo, $(if ($ok) { "IDENTICAL" } else { "DIFFERS" }))
}
