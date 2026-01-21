#!/usr/bin/env python
"""
Wiggum - Tool-Based PRD Task Management System
Runs AI agents with controlled tool-based access to PRD tasks.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Deque, Union, List, Dict, Any, Tuple
from time import sleep

# Global state for signal handling
exit_requested = False
current_process = None
sigint_count = 0

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
    "opencode": ["opencode", "run", "{prompt}"],
    "claude-code": [
        "claude",
        "--permission-mode",
        "bypassPermissions",
        "-p",
        "{prompt}",
    ],
}


@dataclass
class IterationResult:
    """Result of running a single iteration."""

    completed: bool = True
    stopped_early: bool = False
    reason: Optional[str] = None  # "repetition", "tool_call", "timeout", "error"
    exit_code: Optional[int] = None
    tool_call: Optional[str] = None  # "start", "complete", "failed"
    task_id: Optional[str] = None


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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Wiggum - Tool-Based PRD Task Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python wiggum.py --prd-path /path/to/prd.json
  python wiggum.py --prd-path ../prd.json --runner claude-code
  python wiggum.py --prd-path ~/project/prd.json --max-iterations 5 --timeout 300
        """,
    )

    parser.add_argument(
        "--prd-path",
        type=str,
        required=True,
        help="Path to prd.json file (must be OUTSIDE working directory)",
    )

    parser.add_argument(
        "-r",
        "--runner",
        choices=list(RUNNER_TEMPLATES.keys()),
        default="crush",
        help="Runner to use (default: crush)",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for each iteration (default: none)",
    )

    parser.add_argument(
        "-m",
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations before stopping (default: unlimited)",
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
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging",
    )

    return parser.parse_args()


def print_ascii_art():
    """Print the ASCII art header."""
    for line in ASCII_HEADER.split("\n"):
        print(line, flush=True)
        sleep(0.02)

    print(flush=True)


# ============================================================================
# PRD Management Functions
# ============================================================================


def load_prd(path: str) -> List[Dict[str, Any]]:
    """
    Load and parse the PRD JSON file.

    Args:
        path: Path to prd.json file

    Returns:
        List of task dictionaries

    Raises:
        FileNotFoundError: If PRD file doesn't exist
        json.JSONDecodeError: If PRD file is invalid JSON
    """
    with open(path, "r") as f:
        prd = json.load(f)

    # Handle different PRD structures (tasks, features, items)
    tasks = prd.get("tasks", prd.get("features", prd.get("items", [])))

    if not isinstance(tasks, list):
        raise ValueError(f"PRD must contain a 'tasks', 'features', or 'items' list")

    return tasks


def filter_incomplete_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter tasks to only include incomplete ones.

    Args:
        tasks: List of task dictionaries

    Returns:
        List of incomplete task dictionaries
    """
    incomplete = []

    for task in tasks:
        if not isinstance(task, dict):
            continue

        # Check various status field names (passes, passing, status, done)
        status = task.get(
            "passes",
            task.get("passing", task.get("status", task.get("done", False))),
        )

        # Task is incomplete if status is False or not a passing value
        is_passing = (
            status is True
            or status == "passing"
            or status == "done"
            or status == "complete"
        )

        if not is_passing:
            incomplete.append(task)

    return incomplete


def update_task_status(path: str, task_id: str, passing: bool) -> bool:
    """
    Update a task's status in the PRD file.

    Args:
        path: Path to prd.json file
        task_id: ID of the task to update
        passing: True to mark as passing, False otherwise

    Returns:
        True if task was found and updated, False otherwise
    """
    try:
        # Read current PRD
        with open(path, "r") as f:
            prd = json.load(f)

        # Get tasks array
        tasks = prd.get("tasks", prd.get("features", prd.get("items", [])))

        # Find and update task
        task_found = False
        for task in tasks:
            if isinstance(task, dict) and task.get("id") == task_id:
                # Update the status field (prefer "passing", but update others too)
                task["passing"] = passing
                task["passes"] = passing
                task_found = True
                break

        if not task_found:
            return False

        # Write back atomically (write to temp, then rename)
        temp_path = path + ".tmp"
        with open(temp_path, "w") as f:
            json.dump(prd, f, indent=2)

        # Atomic rename
        os.replace(temp_path, path)

        return True

    except Exception as e:
        print(f"\n[Wiggum] Error updating task status: {e}", file=sys.stderr)
        return False


def get_prd_stats(path: str) -> Tuple[int, int, int]:
    """
    Read prd.json and count passing/not passing tasks.

    Args:
        path: Path to prd.json file

    Returns:
        Tuple of (passing, not_passing, total)
    """
    if not os.path.exists(path):
        return (0, 0, 0)

    try:
        tasks = load_prd(path)

        passing = 0
        not_passing = 0

        for task in tasks:
            if isinstance(task, dict):
                # Check various status field names
                status = task.get(
                    "passes",
                    task.get("passing", task.get("status", task.get("done", False))),
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
    except (json.JSONDecodeError, IOError, ValueError):
        return (0, 0, 0)


def get_category_breakdown(path: str) -> Dict[str, Tuple[int, int]]:
    """
    Read prd.json and group tasks by category.

    Args:
        path: Path to prd.json file

    Returns:
        Dict mapping category -> (passing, total)
    """
    if not os.path.exists(path):
        return {}

    try:
        tasks = load_prd(path)

        categories: Dict[str, Tuple[int, int]] = {}

        for task in tasks:
            if isinstance(task, dict):
                category = task.get("category", "uncategorized")
                status = task.get(
                    "passes",
                    task.get("passing", task.get("status", task.get("done", False))),
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
                categories[category] = (
                    passing + (1 if is_passing else 0),
                    total + 1,
                )

        return categories
    except (json.JSONDecodeError, IOError, ValueError):
        return {}


def all_tasks_complete(path: str) -> bool:
    """
    Check if all tasks in the PRD are complete.

    Args:
        path: Path to prd.json file

    Returns:
        True if all tasks are passing, False otherwise
    """
    passing, not_passing, total = get_prd_stats(path)

    # All complete if there are tasks and none are incomplete
    return total > 0 and not_passing == 0


def validate_paths(prd_path: str) -> None:
    """
    Validate that paths are correct and PRD is not in working directory.

    Args:
        prd_path: Path to prd.json file

    Raises:
        FileNotFoundError: If PRD doesn't exist
        ValueError: If PRD is in current working directory
    """
    # Check if PRD exists
    if not os.path.exists(prd_path):
        raise FileNotFoundError(f"PRD file not found: {prd_path}")

    # Get absolute paths
    prd_abs = os.path.abspath(prd_path)
    cwd_abs = os.path.abspath(os.getcwd())

    # Check if PRD is in working directory
    if os.path.dirname(prd_abs) == cwd_abs:
        raise ValueError(
            f"PRD file must be OUTSIDE the working directory.\n"
            f"PRD location: {prd_abs}\n"
            f"Working directory: {cwd_abs}"
        )

    # Try to load and validate PRD structure
    try:
        load_prd(prd_path)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in PRD file: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid PRD structure: {e}")


# ============================================================================
# System Prompt Builder
# ============================================================================


def write_filtered_prd(
    incomplete_tasks: List[Dict[str, Any]], working_dir: str = "."
) -> None:
    """
    Write a filtered PRD file containing only incomplete tasks to the working directory.

    Args:
        incomplete_tasks: List of incomplete task dictionaries
        working_dir: Working directory to write prd.json to (default: current directory)
    """
    prd_path = os.path.join(working_dir, "prd.json")

    # Create a minimal PRD structure with only incomplete tasks
    filtered_prd = {"tasks": incomplete_tasks}

    # Write to working directory
    with open(prd_path, "w") as f:
        json.dump(filtered_prd, f, indent=2)


def build_system_prompt() -> str:
    """
    Build the system prompt with tool instructions.

    Returns:
        Formatted prompt string
    """
    prompt = """You are working on implementing tasks from a Product Requirements Document (PRD).

PRD LOCATION:
The incomplete tasks are in ./prd.json in the current working directory.
Read this file to see which tasks need to be completed.

TOOL INTERFACE:
You have access to two tools for managing task workflow:

1. <complete>{task_id}</complete>
   - Mark a task as complete after successful implementation
   - Use this after you have fully implemented and tested the task
   - This will update the PRD and end the current iteration

2. <failed>{task_id}</failed>
   - Mark a task as failed if you cannot complete it
   - Use this when you encounter blockers or errors
   - This will end the current iteration without updating the task status

PROGRESS LOGGING:
Use ./progress.txt to log progress. Create the file if it doesn't exist.
Append your findings, blockers, and notes as you work on tasks.

WORKFLOW:
1. Read ./prd.json to see the incomplete tasks
2. Select the highest priority task
3. Work on implementing that task
4. Log your progress to ./progress.txt
5. When complete, run git add --all and git commit with an appropriate message, then output <complete>{task_id}</complete>
6. If you cannot complete it, output <failed>{task_id}</failed> with notes in progress.txt for the next person who works in the codebase

Begin by reading the PRD and selecting a task to work on."""

    return prompt


# ============================================================================
# Tool Call Detection
# ============================================================================


def check_for_tool_call(line: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Check if line contains a tool call marker.

    Args:
        line: Line of text to check

    Returns:
        Tuple of (found, tool_type, task_id) where:
        - found: True if tool call detected
        - tool_type: "complete" or "failed" (or None)
        - task_id: The task ID from the tool call (or None)
    """
    stripped = line.strip()

    # Use regex to detect tool calls
    # Pattern: <(complete|failed)>([^<]+)</\1>
    pattern = r"<(complete|failed)>([^<]+)</\1>"
    match = re.search(pattern, stripped)

    if match:
        tool_type = match.group(1)
        task_id = match.group(2).strip()
        return (True, tool_type, task_id)

    return (False, None, None)


# ============================================================================
# Iteration Runner
# ============================================================================


def run_iteration(
    runner: str,
    prompt: str,
    timeout: Optional[int],
    detector: RepetitionDetector,
    prd_path: str,
    verbose: bool = False,
) -> IterationResult:
    """
    Run a single iteration of the AI agent.

    Args:
        runner: Name of the runner to use
        prompt: Prompt to send to the runner
        timeout: Timeout in seconds (None for no timeout)
        detector: RepetitionDetector instance
        prd_path: Path to PRD file (for updating task status)
        verbose: Enable verbose logging

    Returns:
        IterationResult with completion status and tool call info
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

    cmd = cmd.replace("\n", " \\n ")

    try:
        import platform

        # Prepare subprocess arguments to prevent CTRL+C propagation
        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,  # Line buffered - more reliable on Windows
            "shell": True,
        }

        # Platform-specific handling to isolate subprocess from parent's signals
        if platform.system() == "Windows":
            # CREATE_NEW_PROCESS_GROUP prevents child from receiving parent's CTRL+C
            popen_kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        else:
            # On Unix, start a new session so subprocess doesn't receive parent's signals
            popen_kwargs["start_new_session"] = True

        # Start subprocess
        process = subprocess.Popen(cmd, **popen_kwargs)

        # Set global reference for signal handler
        global current_process
        current_process = process

        start_time = time.time()

        # Shared state for the output reading thread
        stop_reason: List[Union[str, None]] = [None]  # Use list for mutable reference
        tool_call_info: List[Tuple[Optional[str], Optional[str]]] = [
            (None, None)
        ]  # (tool_type, task_id)
        thread_done = threading.Event()

        def read_output_worker():
            """Background thread that reads and processes subprocess output."""
            import select

            current_line = ""

            try:
                while True:
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

                    # Read output character by character
                    if platform.system() == "Windows":
                        try:
                            char = process.stdout.read(1)
                            if not char:
                                continue

                            sys.stdout.write(char)
                            sys.stdout.flush()
                            current_line += char

                            if char == "\n":
                                # Check for tool call
                                found, tool_type, task_id = check_for_tool_call(
                                    current_line
                                )
                                if found:
                                    # Update PRD status for complete
                                    if tool_type == "complete":
                                        if verbose:
                                            print(
                                                f"\n[Wiggum] Updating PRD: marking {task_id} as passing",
                                                file=sys.stderr,
                                            )
                                        update_task_status(prd_path, task_id, True)
                                    elif verbose:
                                        print(
                                            f"\n[Wiggum] Task {task_id} failed, keeping status as not passing",
                                            file=sys.stderr,
                                        )

                                    # Stop iteration
                                    tool_call_info[0] = (tool_type, task_id)
                                    stop_reason[0] = f"tool_call:{tool_type}"
                                    process.terminate()
                                    break

                                # Check for repetition
                                if detector.add_line(current_line):
                                    stop_reason[0] = "repetition"
                                    process.terminate()
                                    break

                                current_line = ""
                        except Exception:
                            if process.poll() is not None:
                                break
                    else:
                        # Unix systems
                        ready, _, _ = select.select([process.stdout], [], [], 0.1)
                        if ready:
                            char = process.stdout.read(1)
                            if not char:
                                break

                            sys.stdout.write(char)
                            sys.stdout.flush()
                            current_line += char

                            if char == "\n":
                                # Check for tool call
                                found, tool_type, task_id = check_for_tool_call(
                                    current_line
                                )
                                if found:
                                    # Update PRD status for complete
                                    if tool_type == "complete":
                                        if verbose:
                                            print(
                                                f"\n[Wiggum] Updating PRD: marking {task_id} as passing",
                                                file=sys.stderr,
                                            )
                                        update_task_status(prd_path, task_id, True)
                                    elif verbose:
                                        print(
                                            f"\n[Wiggum] Task {task_id} failed, keeping status as not passing",
                                            file=sys.stderr,
                                            )

                                    tool_call_info[0] = (tool_type, task_id)
                                    stop_reason[0] = f"tool_call:{tool_type}"
                                    process.terminate()
                                    break

                                if detector.add_line(current_line):
                                    stop_reason[0] = "repetition"
                                    process.terminate()
                                    break

                                current_line = ""
            finally:
                thread_done.set()

        # Start the output reading thread
        output_thread = threading.Thread(target=read_output_worker, daemon=True)
        output_thread.start()

        # Main thread monitors for timeout and completion (responsive to signals)
        while not thread_done.is_set():
            # Sleep briefly to allow responsive signal handling
            time.sleep(0.1)

            # Check for timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                process.terminate()
                thread_done.wait(timeout=5)  # Wait for thread to finish
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                current_process = None
                return IterationResult(stopped_early=True, reason="timeout")

            # Check if output thread detected a stop condition
            if stop_reason[0] is not None:
                thread_done.wait(timeout=5)  # Wait for thread to finish
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                current_process = None

                # Extract tool call info
                tool_type, task_id = tool_call_info[0]
                return IterationResult(
                    stopped_early=True,
                    reason=stop_reason[0],
                    tool_call=tool_type,
                    task_id=task_id,
                )

        # Thread is done, wait for process to complete
        output_thread.join(timeout=5)

        # Wait for process to complete
        exit_code = process.wait()

        # Clear global process reference
        current_process = None
        return IterationResult(completed=True, exit_code=exit_code)

    except FileNotFoundError:
        current_process = None
        print(f"\nError: Runner '{runner}' not found in PATH", file=sys.stderr)
        return IterationResult(completed=False, stopped_early=True, reason="error")
    except Exception as e:
        current_process = None
        print(f"\nError running iteration: {e}", file=sys.stderr)
        return IterationResult(completed=False, stopped_early=True, reason="error")


# ============================================================================
# Status Display and Utilities
# ============================================================================


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
    max_iterations: Optional[int],
    prev_iteration_time: Optional[float],
    session_start_time: float,
    stop_reasons: Dict[str, int],
    completed_iterations: int,
    prd_path: str,
    current_task: Optional[str] = None,
) -> None:
    """Print a status bar at the start of each iteration."""
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    session_elapsed = time.time() - session_start_time

    # Get prd.json stats
    passing, not_passing, total = get_prd_stats(prd_path)
    categories = get_category_breakdown(prd_path)
    commits = get_commits_since(session_start_time)

    # Build status bar
    max_iter_str = str(max_iterations) if max_iterations else "∞"
    print(f"\n{'=' * 80}")
    print(f"  ITERATION {iteration}/{max_iter_str}")
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

    # Row 3: Current task
    if current_task:
        print(f"  🎯 Current Task: {current_task}")

    # Row 4: Category breakdown (if categories exist)
    if categories:
        cat_parts = []
        for cat, (cat_pass, cat_total) in sorted(categories.items()):
            marker = "✓" if cat_pass == cat_total else ""
            cat_parts.append(f"{cat}: {cat_pass}/{cat_total}{marker}")
        print(f"  📁 Categories: {' | '.join(cat_parts)}")

    # Row 5: Session stats (commits, completion rate)
    completion_rate = 0
    if completed_iterations > 0:
        completion_rate = (stop_reasons.get("complete", 0) / completed_iterations) * 100
    session_info = f"  📈 Session: {completed_iterations} iterations"
    session_info += f" │ Commits: {commits}"
    session_info += f" │ Rate: {completion_rate:.0f}% complete"
    print(session_info)

    # Row 6: Stop reasons tally (only show if we have completed iterations)
    if completed_iterations > 0:
        stop_parts = []
        for reason in [
            "complete",
            "failed",
            "timeout",
            "repetition",
            "error",
            "normal",
        ]:
            count = stop_reasons.get(reason, 0)
            if count > 0:
                stop_parts.append(f"{count} {reason}")
        if stop_parts:
            print(f"  🏁 Stops: {', '.join(stop_parts)}")

    print(f"{'=' * 80}\n")


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children (entire process tree)."""
    import platform

    try:
        if platform.system() == "Windows":
            # On Windows, use taskkill with /T flag to kill the entire process tree
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
            )
        else:
            # On Unix, kill the process group
            os.killpg(os.getpgid(pid), signal.SIGKILL)  # type: ignore[attr-defined]
    except Exception:
        pass


def handle_sigint(signum, frame):
    """
    Handle SIGINT (CTRL+C) signals.
    First press: Request graceful exit after current iteration.
    Second press: Force immediate exit.
    """
    global exit_requested, current_process, sigint_count

    sigint_count += 1

    if sigint_count == 1:
        exit_requested = True
        print("\n[Wiggum] Exit requested. Will exit after current iteration completes.")
        print("[Wiggum] Press CTRL+C again to exit immediately.")
    else:
        print("\n[Wiggum] Force exit requested. Terminating immediately...")
        if current_process is not None:
            try:
                kill_process_tree(current_process.pid)
            except Exception:
                pass
        sys.exit(130)


# ============================================================================
# Main Loop
# ============================================================================


def main():
    """Main entry point."""
    global exit_requested

    args = parse_args()

    # Validate paths
    try:
        validate_paths(args.prd_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_ascii_art()

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, handle_sigint)

    # Print configuration information
    print(f"{'=' * 60}")
    print("Wiggum Configuration")
    print(f"{'=' * 60}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"PRD Path:          {args.prd_path}")
    print(f"Runner:            {args.runner}")
    print(
        f"Max Iterations:    {args.max_iterations if args.max_iterations else 'unlimited'}"
    )
    print(f"Timeout:           {args.timeout if args.timeout else 'none'}")
    print(f"Rep. Threshold:    {args.repetition_threshold}")
    print(f"History Size:      {args.history_size}")
    print(f"Verbose:           {args.verbose}")
    print(f"{'=' * 60}\n")

    # Initialize repetition detector
    detector = RepetitionDetector(
        history_size=args.history_size, threshold=args.repetition_threshold
    )

    # Track session stats
    session_start_time = time.time()
    prev_iteration_time: Optional[float] = None
    completed_iterations = 0
    stop_reasons: Dict[str, int] = {
        "complete": 0,
        "failed": 0,
        "timeout": 0,
        "repetition": 0,
        "error": 0,
        "normal": 0,
    }
    current_task: Optional[str] = None

    # Main loop - run until all tasks complete
    iteration = 0
    while not all_tasks_complete(args.prd_path):
        iteration += 1

        # Check for user exit request
        if exit_requested:
            print("\n[Wiggum] Exiting gracefully - not starting next iteration")
            break

        # Check max iterations
        if args.max_iterations and iteration > args.max_iterations:
            print(f"\n[Wiggum] Reached maximum iterations ({args.max_iterations})")
            break

        # Print status bar at start of each iteration
        print_status_bar(
            iteration=iteration,
            max_iterations=args.max_iterations,
            prev_iteration_time=prev_iteration_time,
            session_start_time=session_start_time,
            stop_reasons=stop_reasons,
            completed_iterations=completed_iterations,
            prd_path=args.prd_path,
            current_task=current_task,
        )

        # Load and filter PRD
        try:
            tasks = load_prd(args.prd_path)
            incomplete = filter_incomplete_tasks(tasks)

            if not incomplete:
                print("\n[Wiggum] All tasks complete!")
                break

            if args.verbose:
                print(
                    f"[Wiggum] Found {len(incomplete)} incomplete tasks",
                    file=sys.stderr,
                )

        except Exception as e:
            print(f"\n[Wiggum] Error loading PRD: {e}", file=sys.stderr)
            break

        # Write filtered PRD to working directory
        try:
            write_filtered_prd(incomplete)
            if args.verbose:
                print(
                    f"[Wiggum] Wrote {len(incomplete)} incomplete tasks to ./prd.json",
                    file=sys.stderr,
                )
        except Exception as e:
            print(f"\n[Wiggum] Error writing filtered PRD: {e}", file=sys.stderr)
            break

        # Build prompt (tasks are now in ./prd.json)
        prompt = build_system_prompt()

        # Run iteration
        iteration_start_time = time.time()

        result = run_iteration(
            runner=args.runner,
            prompt=prompt,
            timeout=args.timeout,
            detector=detector,
            prd_path=args.prd_path,
            verbose=args.verbose,
        )

        # Track iteration time
        prev_iteration_time = time.time() - iteration_start_time
        completed_iterations += 1

        # Handle results and track stop reasons
        if result.stopped_early:
            if result.reason == "repetition":
                stop_reasons["repetition"] += 1
                print("\n[Wiggum] Repetition detected, moving to next iteration\n")
            elif result.reason and result.reason.startswith("tool_call:"):
                tool_type = result.reason.split(":", 1)[1]
                if tool_type == "complete":
                    stop_reasons["complete"] += 1
                    current_task = result.task_id
                    print(
                        f"\n[Wiggum] Task {result.task_id} completed successfully, moving to next iteration\n"
                    )
                elif tool_type == "failed":
                    stop_reasons["failed"] += 1
                    current_task = None
                    print(
                        f"\n[Wiggum] Task {result.task_id} failed, moving to next iteration\n"
                    )
            elif result.reason == "timeout":
                stop_reasons["timeout"] += 1
                print("\n[Wiggum] Timeout reached, moving to next iteration\n")
            elif result.reason == "error":
                stop_reasons["error"] += 1
                print("\n[Wiggum] Error occurred, stopping execution\n")
                sys.exit(1)
        else:
            stop_reasons["normal"] += 1

        # Reset detector for next iteration
        detector.reset()

    # Final status
    if all_tasks_complete(args.prd_path):
        print(f"\n{'=' * 60}")
        print("All tasks in PRD are complete!")
        print(f"{'=' * 60}\n")
    else:
        print(f"\n{'=' * 60}")
        print(f"Completed {iteration} iteration(s)")
        passing, not_passing, total = get_prd_stats(args.prd_path)
        print(f"PRD Status: {passing}/{total} tasks passing, {not_passing} remaining")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
