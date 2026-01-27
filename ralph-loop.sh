#!/bin/bash
# ralph-loop.sh
# Usage: ralph-loop.sh <iterations>
# Loops up to <iterations> times or until <promise>COMPLETE</promise> is received

set -e

# ============================================
# CONFIGURATION: Set your preferred runner
# ============================================
# Options: "claude-code", "opencode", "crush", "nanocoder"
RUNNER="crush"

# Check if iterations argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <iterations>" >&2
  exit 1
fi

max_iterations=$1

# Validate that iterations is a positive integer
if ! [[ "$max_iterations" =~ ^[0-9]+$ ]] || [ "$max_iterations" -le 0 ]; then
  echo "Error: iterations must be a positive integer" >&2
  exit 1
fi

echo "Starting ralph loop: max $max_iterations iterations" >&2
echo "Will stop early if <promise>COMPLETE</promise> is detected" >&2
echo "========================================" >&2

for ((i=1; i<=max_iterations; i++)); do
  echo "" >&2
  echo "=== Iteration $i/$max_iterations ===" >&2
  echo "" >&2

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
If, while implementing the feature, you notice the PRD is complete, output <promise>COMPLETE</promise>.
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

  # Check if COMPLETE marker was found
  if echo "$result" | grep -q "<promise>COMPLETE</promise>"; then
    echo "" >&2
    echo "========================================" >&2
    echo "COMPLETE marker detected after iteration $i" >&2
    echo "Stopping loop early." >&2
    exit 0
  fi
done

echo "" >&2
echo "========================================" >&2
echo "Completed all $max_iterations iterations" >&2
