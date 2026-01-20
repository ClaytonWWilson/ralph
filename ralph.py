#!/usr/bin/env python
"""
Ralph - Advanced AI Agent Runner
Runs AI agents (crush, opencode, claude-code) with iteration control, timeouts, and repetition detection.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Deque
import os
from time import sleep

ASCII_HEADER = """----------------------------------------------------------------------------------------------------=
--------------------------------------------------==------------------------------------------------=
=---------------------------------=------+++***#######**++==-----------------------------------------
----------------------------------=+*#%%#%%*:+%+ *%#=#=.+@%####*+=----------------------------------=
------------------------------+*#%#*#%= :*: :*. :* .=.*  .#+   *%%###+---=--------------------------=
=--------------------------+#%@@@-.##  *+  #*  -*   = :#   +*    #=  #@#+--=-------------------------
-------------------------*%%**%+ =%= =#: .#=   %.   .: -+   -*    =*  =%@%*+=-----------------------=
---------------------=*%@*  =#  **  *=   %:   %:     =  +.    #     =   =+ #@#+----------------------
=------------------+#@@@=  ##  ## .#=   #    +*      +  :+    :+     +   == =@@#=-=------------------
-----------------*%%@%=  -%: :%- +%    #-    #:      =   #     :-     *   .+ :@@@%*==---------------=
--------------=*#*#@%:  -#   %: -*    =+     :      .    =      .      :   -  :%@@@%=----------------
=-----------=*#++#%#   ++   #. =*     =                                        -#*###*=--------------
-----------+#*=*%@#   *=   *= .*                .                               **=#+**-------------=
=---------+*==##%%   +-   %-  #            =#=.     -+:              =+***=     -%==*+==-------------
=-----------=*+*@=  ==   =+   .          =#.          .#.          *=      +*    #*-+*---------------
------------=-+%+  ++    *              :=              =.       :%          #+  #+--=--------------=
=-------------#*  =*    :=              *      .%@.      +       ==           # :#+------------------
----------=*##%#=:*     .               *                =       -*      *%#  # +#=------------------
---------=#+    :#:                     :=              +         #-         -* **------------------=
=--------++                              +#           :*           -+       ==  ##+------------------
--------=*=                                -#*-.  .=*+       +**###*=:-+*++.    + .#+----------------
---------=#.                                   ...                  -#-        :=  -+---------------=
=---------+#=    :                                                    #=        *  -*----------------
-----------=+*%%-                                                     #*        :#=#+---------------=
--------------+%:                                                   :##           *#=---------------=
=-------------=#*                                            *#**#%*-              -#--=-------------
---------------=%=                         .--:                                     *=--------------=
----------------+%.                      #-     :***-                              =*=--------------=
=----------------+%=                   .%@#:  :*=      =**+-.          .::-====+*#*=-----------------
------------------+%*                  =@@@@@@@@@#.    =-    ..:--::.  **===-=----------------------=
=-----------------*%%%+                .%@@@@@@@@@@@@@@@@@@%##%%@@#=:=#*-----------------------------
=----------------=%*-:*%#               -@@@@@@@@@@@@@@@@@@@@@@@@%   :%=-----------------------------
-----------------*#-::.:-**=.            :#@%=--=*%@%#%@@@@@@@@@@*    %+==--------------------------=
=---------------=#+:::.....=*%%+.           =%*-:--*=::-+%@@@@@@@*    -%**##*=-----------------------
=----------------#+:::........:=*%%*-         .*%*+-:::::-#@@@@@@@:    -*.   *#=---------------------
-------------=*#%@@#-..............:=*%@%*-        -+#%%#**%@@@@@@%     ++#*-**#*-------------------=
=-------==-=%%+-::-+%+.............. ...:-+*#%%%#+=:                   +-  .%*  +#=------------------
=-------=+%#=:::::...+%+:.........................::*##*.           .:     #.-#= :#+-----------------
-------=#%+:::::.....:#@%=.........................+#:.:+#:              -%:   *= .*+---------------=
=-----+%#-::::........=%+=#+:....................:*#:.....:=*=          += ...  +#- ++---------------
=----=%#-:::......... :#*..-*+:.................:%#:.........-*#=     :#-....... -%++*-:-------------
----=##-:::...........:*#... :+%*:.............:#+:..............:=*##+.........   +#==-------------=
=---#%-:::............:*#.......:*%*-.........-%*..................... ...........  *#=--------------
*****************************************************************************************************
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%*  %*%%%%%%@-  =+@%%%%%%%%%%%%%%%%%%%%#  #%%%%%  %%%%%%%%%%%%%%%%%@  =@%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%=  ####*+*#*+   *+=%%%#*%%%#*%%%%#*#%%%: -%%#*%+  %%%#**%%%%%#**%%%%   %%%%%%%%%%%%%%
####%%%%%%%%%%%#  %%%.     =%       +%  +%*  #@      #@  #+  =%  +@:     :@+      @%=  %%%%%%%%%%%%%%
#%%%%%####%%%%%.  @%. :%@+.+@  *@#  ++  #%=  #  :%@*:*%     *%@  %.  -=:  @.  -+#%%%:  %%%%%%%%%%%%%%
######%@@%####+  =%#  %%@+=%*  %%=  *  :%%  .#  *@@+=@.     *%*  @  :##**#@##+   +@@   @%%%%%%%%%%%%%
##########%%@@+  =%%:     -%. =%%  .%       *%      :#  =%   @   @:      *#      #@*  *@%%%%%%%%%%%%%
##############%-  @%%%**#%%%##%%%##%%%#*#%##%%%%#*#%%%##%%%##%##%%%%#*#%%%%%#**#%%*  %%%%%%%%%%%%%%%%
##############%%::%%%%%%%##########%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%-:#%%%%%%%%%%%%%%%%
############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#############################%@@%#################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#####%%+*%#+%################*  +%%@@%%###########%**%###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#####%.  %  %%#%%#%%%########* .*##%%%%@@%%######%%  @########%%%%%%%%%%%%%%%%%%%%%%%#%%%%%%%%%%%%%%%
#####*  -% :#      .    %%###  .#       ###%%#       @#      %%      -%%:      +%:     #%     +%%%%%%
####%+  #%%%+  *%.  %+  *%##%  +#  +%*  +%##%  :%#   %*-*#=  @*  #%=  @.  %@   #   ##  :#  -%@@@%%%%#
####%-  #%%%: -%#  #%=  %%#%%  %+  %%-  #%#%+  #%#  **       @   %%. :@  +%%  :#       +=  %%%%%%%%%#
#####  -%%%%  *%= .%%. .%%#%*  %. =%%  -#%%#*  .=   %   %=   @  :%@  *@   :   #%   #.  %   %%   %%%%#
#####=-%%##%--#%#-#%%+:####%*-#%+-%%#--%%###%#. =*:+%%- .*=:*%--#%@-+%##*=+*  %%%=. :%%%=-*%%-:*%%%%#
###################################################################%@@#      ########################
#####################################################################################################
#####################################################################################################
"""


# Runner command templates
RUNNER_TEMPLATES = {
    "crush": ["crush", "run", "-d", "{prompt}"],
    "nanocoder": ["nanocoder", "run", "{prompt}"],
    "opencode": ["opencode", "run", "--directive", "{prompt}"],
    "claude-code": ["claude", "code", "-d", "{prompt}"],
}


# Default prompt template
DEFAULT_PROMPT_TEMPLATE = "{file_refs} \
1. Find the highest-priority feature that is not already passing to work on and work only on that feature. \
This should be the one YOU decide has the highest priority - not necessarily the first in the list. \
2. Append to progress.txt consistently with your progress updates and findings while implementing the feature. \
Use this to leave a note for the next person working in the codebase. \
3. Update the prd.json with the work that was done. \
4. Make a git commit of that feature using `git add --all` and `git commit -m <message>`. \
ONLY WORK ON A SINGLE FEATURE. \
If you are unable to complete a feature or get stuck on a failing test, save your findings in progress.txt and output <promise>IN PROGRESS</promise>. \
5. After implementing the feature, output <promise>COMPLETE</promise> to signal it is complete. \
If, while implementing the feature, you notice that it's already complete, output <promise>COMPLETE</promise>. \
Only mark a single feature as passing, then output <promise>COMPLETE</promise>."


@dataclass
class IterationResult:
    """Result of running a single iteration."""

    completed: bool = True
    stopped_early: bool = False
    reason: Optional[str] = None  # "repetition", "promise", "timeout", "error"
    exit_code: Optional[int] = None


class RepetitionDetector:
    """Detects repeated sentences in streaming output."""

    def __init__(self, history_size: int = 50, threshold: int = 2):
        """
        Initialize the repetition detector.

        Args:
            history_size: Number of recent sentences to keep in history (default: 50)
            threshold: Number of times a sentence must repeat to trigger detection (default: 2)
        """
        self.history_size = history_size
        self.threshold = threshold
        self.sentence_history: Deque[str] = deque(maxlen=history_size)
        self.sentence_counts: dict = {}  # Track how many times each sentence appears

    def add_line(self, line: str) -> bool:
        """
        Add a line to the detector and check for repetition.
        Returns True if repetition threshold exceeded.
        """
        # Extract sentences from the line
        sentences = self._extract_sentences(line)

        for sentence in sentences:
            # Check if this sentence already exists in recent history
            if sentence in self.sentence_history:
                # Increment count for this sentence
                self.sentence_counts[sentence] = (
                    self.sentence_counts.get(sentence, 1) + 1
                )

                # Check if threshold exceeded
                if self.sentence_counts[sentence] >= self.threshold:
                    return True
            else:
                # New sentence - add to history and initialize count
                self.sentence_history.append(sentence)
                self.sentence_counts[sentence] = 1

                # Clean up counts for sentences that have fallen out of history
                # Keep only sentences that are still in the deque
                if len(self.sentence_counts) > self.history_size:
                    sentences_in_history = set(self.sentence_history)
                    self.sentence_counts = {
                        s: c
                        for s, c in self.sentence_counts.items()
                        if s in sentences_in_history
                    }

        return False

    def _extract_sentences(self, line: str) -> list[str]:
        """Extract and normalize sentences from a line."""
        sentences = []
        line = line.strip()

        # Skip very short lines
        if len(line) < 10:
            return sentences

        # Extract sentences (split by period, question mark, exclamation)
        raw_sentences = re.split(r"[.!?]+", line)
        for sentence in raw_sentences:
            sentence = sentence.strip()
            # Only keep sentences with at least 10 characters
            if len(sentence) >= 10:
                # Normalize whitespace
                normalized = " ".join(sentence.split())
                sentences.append(normalized)

        return sentences

    def reset(self):
        """Reset detector state for next iteration."""
        self.sentence_history.clear()
        self.sentence_counts.clear()


def check_for_promise(line: str) -> tuple[bool, str]:
    """
    Check if line contains a promise marker.
    Returns (found, type) where type is 'complete', 'in_progress', or None.
    """
    # Check for completion marker
    if "<promise>COMPLETE</promise>" == line.strip():
        return (True, "complete")

    # Check for in-progress marker
    if "<promise>IN PROGRESS</promise>" == line.strip():
        return (True, "in_progress")

    # Check for common patterns where AI mentions outputting the promise
    # (case-insensitive matching)
    # line_lower = line.lower()
    # promise_mention_patterns = [
    #     "output <promise>complete</promise>",
    #     "outputting <promise>complete</promise>",
    #     "will now output <promise>complete</promise>",
    #     "i will now output <promise>complete</promise>",
    #     "now output <promise>complete</promise>",
    #     "to indicate that the feature is complete",
    #     "indicate completion",
    # ]

    # for pattern in promise_mention_patterns:
    #     if pattern in line_lower:
    #         return True

    return (False, None)


def run_iteration(
    runner: str, prompt: str, timeout: Optional[int], detector: RepetitionDetector
) -> IterationResult:
    """
    Run a single iteration of the AI agent.

    Args:
        runner: Name of the runner to use
        prompt: Prompt to send to the runner
        timeout: Timeout in seconds (None for no timeout)
        detector: RepetitionDetector instance

    Returns:
        IterationResult with completion status and reason
    """

    # Build command from template
    cmd_template = RUNNER_TEMPLATES[runner]
    cmd_list = [
        part.format(prompt=prompt) if "{prompt}" in part else part
        for part in cmd_template
    ]

    # Join command parts into a string for shell execution
    # Properly quote arguments that may contain spaces or special characters
    cmd_parts = []
    for part in cmd_list:
        # If the part contains spaces or newlines, wrap in quotes
        if " " in part or "\n" in part:
            # Escape any existing quotes and wrap in double quotes
            escaped = part.replace('"', '\\"')
            cmd_parts.append(f'"{escaped}"')
        else:
            cmd_parts.append(part)

    cmd = " ".join(cmd_parts)

    try:
        # Start subprocess
        # Use shell=True on Windows to properly resolve commands like 'crush' that have .cmd wrappers
        # We explicitly cd to the working directory in the shell command itself
        # Use line buffering (bufsize=1) which is more reliable than unbuffered on Windows
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered - more reliable on Windows
            shell=True,
        )

        start_time = time.time()
        current_line = ""

        # Use select/poll on Unix-like systems, or just iterate on Windows
        # Read output in real-time
        import select
        import platform

        while True:
            # Check for timeout first
            if timeout is not None and (time.time() - start_time) > timeout:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return IterationResult(stopped_early=True, reason="timeout")

            # Check if process is still running
            if process.poll() is not None:
                # Process has ended, read any remaining output
                remaining = process.stdout.read()
                if remaining:
                    sys.stdout.write(remaining)
                    sys.stdout.flush()
                    for line in remaining.splitlines(keepends=True):
                        if line.endswith("\n"):
                            detector.add_line(line)
                break

            # Try to read with a small timeout to avoid blocking indefinitely
            # On Windows, select doesn't work with pipes, so we use readline
            if platform.system() == "Windows":
                # Read one character at a time on Windows for real-time streaming
                try:
                    char = process.stdout.read(1)
                    if not char:
                        continue

                    # Print character immediately
                    sys.stdout.write(char)
                    sys.stdout.flush()

                    # Build up current line for detection
                    current_line += char

                    # When we hit a newline, process the line
                    if char == "\n":
                        # Check for promise marker
                        found, promise_type = check_for_promise(current_line)
                        if found:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            return IterationResult(
                                stopped_early=True, reason=f"promise:{promise_type}"
                            )

                        # Check for repetition
                        if detector.add_line(current_line):
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            return IterationResult(
                                stopped_early=True, reason="repetition"
                            )

                        # Reset current line
                        current_line = ""

                except Exception:
                    # If read fails, process might have ended
                    if process.poll() is not None:
                        break
            else:
                # On Unix-like systems, use select for better performance
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
                if ready:
                    char = process.stdout.read(1)
                    if not char:
                        break

                    sys.stdout.write(char)
                    sys.stdout.flush()
                    current_line += char

                    if char == "\n":
                        found, promise_type = check_for_promise(current_line)
                        if found:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            return IterationResult(
                                stopped_early=True, reason=f"promise:{promise_type}"
                            )

                        if detector.add_line(current_line):
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            return IterationResult(
                                stopped_early=True, reason="repetition"
                            )

                        current_line = ""

        # Wait for process to complete
        exit_code = process.wait()

        return IterationResult(completed=True, exit_code=exit_code)

    except FileNotFoundError:
        print(f"\nError: Runner '{runner}' not found in PATH", file=sys.stderr)
        return IterationResult(completed=False, stopped_early=True, reason="error")
    except Exception as e:
        print(f"\nError running iteration: {e}", file=sys.stderr)
        return IterationResult(completed=False, stopped_early=True, reason="error")


def build_prompt(file_refs: str, custom_prompt: Optional[str]) -> str:
    """Build the prompt to send to the runner."""
    if custom_prompt:
        return custom_prompt

    return DEFAULT_PROMPT_TEMPLATE.format(file_refs=file_refs)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ralph - Advanced AI Agent Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ralph.py
  python ralph.py --iterations 5 --timeout 300
  python ralph.py --runner opencode --iterations 3
  python ralph.py --prompt "Fix the bug in main.py"
        """,
    )

    parser.add_argument(
        "-r",
        "--runner",
        choices=list(RUNNER_TEMPLATES.keys()),
        default="crush",
        help="Runner to use (default: crush)",
    )

    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=1,
        help="Number of iterations to run (default: 1)",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each iteration (default: none)",
    )

    parser.add_argument(
        "-rt",
        "--repetition-threshold",
        type=int,
        default=5,
        help="Number of times a sentence must repeat to trigger detection (default: 5)",
    )

    parser.add_argument(
        "-hs",
        "--history-size",
        type=int,
        default=50,
        help="Number of recent sentences to check for repetition (default: 50)",
    )

    parser.add_argument(
        "-f",
        "--file-refs",
        type=str,
        default="@prd.json @progress.txt",
        help="File references to include in prompt (default: '@prd.json @progress.txt')",
    )

    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Override the default prompt entirely",
    )

    return parser.parse_args()


def get_prd_stats() -> tuple[int, int, int]:
    """
    Read prd.json and count passing/not passing features.
    Returns (passing, not_passing, total).
    """
    prd_path = os.path.join(os.getcwd(), "prd.json")
    if not os.path.exists(prd_path):
        return (0, 0, 0)

    try:
        with open(prd_path, "r") as f:
            prd = json.load(f)

        passing = 0
        not_passing = 0

        # Handle different prd.json structures (tasks, features, items)
        features = prd.get("tasks", prd.get("features", prd.get("items", [])))
        if isinstance(features, list):
            for feature in features:
                if isinstance(feature, dict):
                    # Check various status field names (passes, passing, status, done)
                    status = feature.get(
                        "passes", feature.get("passing", feature.get("status", feature.get("done", False)))
                    )
                    if (
                        status is True
                        or status == "passing"
                        or status == "done"
                        or status == "complete"
                    ):
                        passing += 1
                    else:
                        not_passing += 1

        return (passing, not_passing, passing + not_passing)
    except (json.JSONDecodeError, IOError):
        return (0, 0, 0)


def get_category_breakdown() -> dict[str, tuple[int, int]]:
    """
    Read prd.json and group tasks by category.
    Returns dict mapping category -> (passing, total).
    """
    prd_path = os.path.join(os.getcwd(), "prd.json")
    if not os.path.exists(prd_path):
        return {}

    try:
        with open(prd_path, "r") as f:
            prd = json.load(f)

        categories: dict[str, tuple[int, int]] = {}

        features = prd.get("tasks", prd.get("features", prd.get("items", [])))
        if isinstance(features, list):
            for feature in features:
                if isinstance(feature, dict):
                    category = feature.get("category", "uncategorized")
                    status = feature.get(
                        "passes", feature.get("passing", feature.get("status", feature.get("done", False)))
                    )
                    is_passing = (
                        status is True
                        or status == "passing"
                        or status == "done"
                        or status == "complete"
                    )

                    if category not in categories:
                        categories[category] = (0, 0)

                    passing, total = categories[category]
                    categories[category] = (passing + (1 if is_passing else 0), total + 1)

        return categories
    except (json.JSONDecodeError, IOError):
        return {}


def get_commits_since(timestamp: float) -> int:
    """
    Count git commits made since the given Unix timestamp.
    Returns 0 if not in a git repo or on error.
    """
    try:
        # Convert timestamp to ISO format for git
        from datetime import datetime as dt
        since_str = dt.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        result = subprocess.run(
            ["git", "rev-list", "--count", f"--since={since_str}", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
        return 0
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        return 0


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def print_status_bar(
    iteration: int,
    total_iterations: int,
    prev_iteration_time: Optional[float],
    session_start_time: float,
    stop_reasons: dict[str, int],
    completed_iterations: int,
) -> None:
    """Print a status bar at the start of each iteration."""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    session_elapsed = time.time() - session_start_time

    # Get prd.json stats
    passing, not_passing, total = get_prd_stats()
    categories = get_category_breakdown()
    commits = get_commits_since(session_start_time)

    # Build status bar
    print(f"\n{'=' * 80}")
    print(f"  ITERATION {iteration}/{total_iterations}")
    print(f"{'=' * 80}")

    # Row 1: Time info
    time_info = f"  🕐 Current Time: {current_time}"
    time_info += f"  │  Session: {format_duration(session_elapsed)}"
    if prev_iteration_time is not None:
        time_info += f"  │  Prev Iteration: {format_duration(prev_iteration_time)}"
    print(time_info)

    # Row 2: PRD stats (if prd.json exists)
    if total > 0:
        progress_pct = (passing / total) * 100 if total > 0 else 0
        bar_width = 20
        filled = int(bar_width * passing / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        print(
            f"  📊 PRD Progress: [{bar}] {passing}/{total} passing ({progress_pct:.0f}%)"
        )
    else:
        print("  📊 PRD Progress: No prd.json found")

    # Row 3: Category breakdown (if categories exist)
    if categories:
        cat_parts = []
        for cat, (cat_pass, cat_total) in sorted(categories.items()):
            marker = "✓" if cat_pass == cat_total else ""
            cat_parts.append(f"{cat}: {cat_pass}/{cat_total}{marker}")
        print(f"  📁 Categories: {' | '.join(cat_parts)}")

    # Row 4: Session stats (commits, completion rate)
    completion_rate = 0
    if completed_iterations > 0:
        completion_rate = (stop_reasons.get("complete", 0) / completed_iterations) * 100
    session_info = f"  📈 Session: {completed_iterations} iterations"
    session_info += f" │ Commits: {commits}"
    session_info += f" │ Rate: {completion_rate:.0f}% complete"
    print(session_info)

    # Row 5: Stop reasons tally (only show if we have completed iterations)
    if completed_iterations > 0:
        stop_parts = []
        for reason in ["complete", "in_progress", "timeout", "repetition", "error", "normal"]:
            count = stop_reasons.get(reason, 0)
            if count > 0:
                stop_parts.append(f"{count} {reason}")
        if stop_parts:
            print(f"  🏁 Stops: {', '.join(stop_parts)}")

    print(f"{'=' * 80}\n")


def print_ascii_art():
    for line in ASCII_HEADER.split("\n"):
        print(line, flush=True)
        sleep(0.02)

    print(flush=True)


def main():
    """Main entry point."""

    args = parse_args()

    print_ascii_art()

    # Print configuration information
    print(f"{'=' * 60}")
    print("Ralph Configuration")
    print(f"{'=' * 60}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Runner:            {args.runner}")
    print(f"Iterations:        {args.iterations}")
    print(f"Timeout:           {args.timeout if args.timeout else 'none'}")
    print(f"Rep. Threshold:    {args.repetition_threshold}")
    print(f"History Size:      {args.history_size}")
    print(f"File Refs:         {args.file_refs}")
    if args.prompt:
        print(
            f"Custom Prompt:     {args.prompt[:50]}{'...' if len(args.prompt) > 50 else ''}"
        )
    print(f"{'=' * 60}\n")

    # Build prompt
    prompt = build_prompt(args.file_refs, args.prompt)

    # Initialize repetition detector
    detector = RepetitionDetector(
        history_size=args.history_size, threshold=args.repetition_threshold
    )

    # Track session stats
    session_start_time = time.time()
    prev_iteration_time: Optional[float] = None
    completed_iterations = 0
    stop_reasons: dict[str, int] = {
        "complete": 0,
        "in_progress": 0,
        "timeout": 0,
        "repetition": 0,
        "error": 0,
        "normal": 0,
    }

    # Run iterations
    for iteration in range(1, args.iterations + 1):
        # Print status bar at start of each iteration
        print_status_bar(
            iteration=iteration,
            total_iterations=args.iterations,
            prev_iteration_time=prev_iteration_time,
            session_start_time=session_start_time,
            stop_reasons=stop_reasons,
            completed_iterations=completed_iterations,
        )

        iteration_start_time = time.time()

        result = run_iteration(
            runner=args.runner, prompt=prompt, timeout=args.timeout, detector=detector
        )

        # Track iteration time
        prev_iteration_time = time.time() - iteration_start_time
        completed_iterations += 1

        # Handle early stops and track stop reasons
        if result.stopped_early:
            if result.reason == "repetition":
                stop_reasons["repetition"] += 1
                print("\n[Ralph] Repetition detected, moving to next iteration\n")
            elif result.reason and result.reason.startswith("promise:"):
                promise_type = result.reason.split(":", 1)[1]
                if promise_type == "complete":
                    stop_reasons["complete"] += 1
                    print(
                        "\n[Ralph] Feature completion detected (<promise>COMPLETE</promise>), moving to next iteration\n"
                    )
                elif promise_type == "in_progress":
                    stop_reasons["in_progress"] += 1
                    print(
                        "\n[Ralph] Work in progress detected (<promise>IN PROGRESS</promise>), moving to next iteration\n"
                    )
                else:
                    stop_reasons["normal"] += 1
                    print(
                        "\n[Ralph] Promise marker detected, moving to next iteration\n"
                    )
            elif result.reason == "timeout":
                stop_reasons["timeout"] += 1
                print("\n[Ralph] Timeout reached, moving to next iteration\n")
            elif result.reason == "error":
                stop_reasons["error"] += 1
                print("\n[Ralph] Error occurred, stopping execution\n")
                sys.exit(1)
        else:
            stop_reasons["normal"] += 1

        # Reset detector for next iteration
        detector.reset()

    print(f"\n{'=' * 60}")
    print(f"Completed {args.iterations} iteration(s)")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
