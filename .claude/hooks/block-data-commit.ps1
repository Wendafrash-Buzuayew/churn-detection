# PreToolUse hook (matcher: Bash). Blocks git add/commit of data/model artifacts
# (*.csv, *.joblib, *.pkl) that must never enter version control (see README /
# .gitignore). Safety net for `git add -f` or files .gitignore doesn't cover yet.

$ErrorActionPreference = 'SilentlyContinue'

$stdin = [Console]::In.ReadToEnd()
try {
    $data = $stdin | ConvertFrom-Json
} catch {
    exit 0
}

$command = $data.tool_input.command
if (-not $command) { exit 0 }

$extPattern = '\.(csv|joblib|pkl)("|''|\s|$)'
$blocked = $false
$reason = $null

if ($command -match '\bgit\s+add\b' -and $command -match $extPattern) {
    $blocked = $true
    $reason = "Blocked: this 'git add' command references a .csv/.joblib/.pkl file. Data snapshots and model artifacts must never be committed (see README + .gitignore). If this is intentional, add the file manually outside Claude Code."
}

if (-not $blocked -and $command -match '\bgit\s+commit\b') {
    $projectDir = $env:CLAUDE_PROJECT_DIR
    if ($projectDir) { Push-Location $projectDir }
    $staged = git diff --cached --name-only 2>$null
    if ($projectDir) { Pop-Location }
    $badFiles = @($staged | Where-Object { $_ -match '\.(csv|joblib|pkl)$' })
    if ($badFiles.Count -gt 0) {
        $blocked = $true
        $reason = "Blocked: staged files include data/model artifacts that must never be committed: $($badFiles -join ', ')"
    }
}

if ($blocked) {
    $result = @{
        systemMessage = $reason
        hookSpecificOutput = @{
            hookEventName = "PreToolUse"
            permissionDecision = "deny"
            permissionDecisionReason = $reason
        }
    } | ConvertTo-Json -Depth 5 -Compress
    Write-Output $result
}

exit 0
