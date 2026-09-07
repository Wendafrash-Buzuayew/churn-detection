# PostToolUse hook (matcher: Edit|Write). When a churn_ranker/<name>.py module is
# edited, auto-run its matching test module for fast TDD feedback (README:
# "Tests follow TDD; every module has hand-computed expected values").

$ErrorActionPreference = 'SilentlyContinue'

$stdin = [Console]::In.ReadToEnd()
try {
    $data = $stdin | ConvertFrom-Json
} catch {
    exit 0
}

$filePath = $data.tool_input.file_path
if (-not $filePath) { $filePath = $data.tool_response.filePath }
if (-not $filePath) { exit 0 }

$normalized = $filePath -replace '\\', '/'
if ($normalized -notmatch '(^|/)churn_ranker/([A-Za-z0-9_]+)\.py$') { exit 0 }

$module = $Matches[2]
if ($module -eq '__init__') { exit 0 }

$projectDir = $env:CLAUDE_PROJECT_DIR
if (-not $projectDir) { $projectDir = (Get-Location).Path }

$testRelPath = "tests/test_$module.py"
$testFullPath = Join-Path $projectDir $testRelPath
if (-not (Test-Path $testFullPath)) { exit 0 }

$python = Join-Path $projectDir "venv/Scripts/python.exe"
if (-not (Test-Path $python)) { exit 0 }

Push-Location $projectDir
$output = & $python -m pytest $testRelPath -q 2>&1 | Out-String
$exitCode = $LASTEXITCODE
Pop-Location

$status = if ($exitCode -eq 0) { "PASSED" } else { "FAILED" }
$context = "Auto-ran ``pytest $testRelPath`` after editing $normalized -> $status`n`n$output"

$result = @{
    systemMessage = "pytest $testRelPath : $status"
    hookSpecificOutput = @{
        hookEventName = "PostToolUse"
        additionalContext = $context
    }
} | ConvertTo-Json -Depth 5 -Compress

Write-Output $result
exit 0
