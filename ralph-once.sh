#!/bin/bash
# ralph.sh
# Usage: ralph.sh

set -e

# ============================================
# CONFIGURATION: Set your preferred runner
# ============================================
# Options: "claude-code", "opencode", "crush", "nanocoder"
RUNNER="crush"

# Define the prompt content
prompt_content=$(cat <<'EOF'
@prd.json @progress.txt
1. Find the highest-priority feature that is not already passing to work on and work only on that feature.
This should be the one YOU decide has the highest priority - not necessarily the first in the list.
2. Update the prd.json with the work that was done.
3. Append to progress.txt consistently with your progress and findings while implementing the prd.
Use this to leave a note for the next person working in the codebase.
4. Make a git commit of that feature using `git commit --all`.
ONLY WORK ON A SINGLE FEATURE.
If you are unable to complete a feature or get stuck on a failing test, save your findings in progress.txt and stop.
EOF
)

# Build and execute the command based on the runner
case "$RUNNER" in
  "claude-code")
    result=$(claude --permission-mode bypassPermissions -p "$prompt_content" | tee /dev/stderr)
    ;;
  "opencode")
    result=$(opencode run -d "$prompt_content" | tee /dev/stderr)
    ;;
  "crush")
    result=$(crush run -d "$prompt_content" | tee /dev/stderr)
    ;;
  "nanocoder")
    result=$(nanocoder run -d "$prompt_content" | tee /dev/stderr)
    ;;
  *)
    echo "Error: Unknown runner '$RUNNER'" >&2
    echo "Valid options: claude-code, opencode, crush, nanocoder" >&2
    exit 1
    ;;
esac

echo "$result"
