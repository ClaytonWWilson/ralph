#!/bin/bash
# ralph.sh
# Usage: ralph.sh

set -e

result=$(
  crush run -d "$(cat <<'EOF'
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
)" | tee /dev/stderr
)



echo "$result"
