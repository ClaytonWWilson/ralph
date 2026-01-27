#!/usr/bin/env python
"""
Ralph MCP - PRD Task Management with MCP Protocol

An orchestrator that runs AI agents iteratively to complete tasks from a PRD file.
Uses MCP (Model Context Protocol) for task management instead of XML tool calls.

This is a reimplementation of wiggum.py using MCP for task lifecycle management.
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
from pathlib import Path
from time import sleep
from typing import Deque, Dict, List, Optional, Tuple

# Global state for signal handling
exit_requested = False
current_process: Optional[subprocess.Popen] = None
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

RUNNER_TEMPLATES = {
    "claude-code": [
        "claude",
        "--permission-mode",
        "bypassPermissions",
        "-p",
        "{prompt}",
    ],
    "opencode": [
        "opencode",
        "run",
        "{prompt}",
    ],
    "crush": [
        "crush",
        "run",
        "{prompt}",
    ],
    "nanocoder": [
        "nanocoder",
        "run",
        "{prompt}",
    ],
}


@dataclass
class IterationResult:
    """Result of running a single iteration."""

    completed: bool = True
    stopped_early: bool = False
    reason: Optional[str] = None  # "task_complete", "task_failed", "timeout", "error"
    exit_code: Optional[int] = None
    task_id: Optional[str] = None


class RepetitionDetector:
    """Detects repeated sentences in streaming output."""

    def __init__(self, history_size: int = 50, threshold: int = 2):
        self.history_size = history_size
        self.threshold = threshold
        self.sentence_history: Deque[str] = deque(maxlen=history_size)
        self.sentence_counts: dict = {}

    def add_line(self, line: str) -> bool:
        """Add a line to the detector. Returns True if repetition threshold exceeded."""
        sentences = self._extract_sentences(line)

        for sentence in sentences:
            if sentence in self.sentence_history:
                self.sentence_counts[sentence] = (
                    self.sentence_counts.get(sentence, 1) + 1
                )
                if self.sentence_counts[sentence] >= self.threshold:
                    return True
            else:
                self.sentence_history.append(sentence)
                self.sentence_counts[sentence] = 1

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

        if len(line) < 10:
            return sentences

        raw_sentences = re.split(r"[.!?]+", line)
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if len(sentence) >= 10:
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
        description="Ralph MCP - PRD Task Management with MCP Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ralph_mcp.py --prd-path /path/to/prd.json
  python ralph_mcp.py --prd-path ../prd.json --runner claude-code
  python ralph_mcp.py --prd-path ~/project/prd.json --max-iterations 5 --timeout 300
        """,
    )

    parser.add_argument(
        "--prd-path",
        type=str,
        required=True,
        help="Path to prd.json file",
    )

    parser.add_argument(
        "-r",
        "--runner",
        choices=list(RUNNER_TEMPLATES.keys()),
        default="claude-code",
        help="Runner to use (default: claude-code)",
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

    parser.add_argument(
        "--print-mcp-config",
        action="store_true",
        help="Print the MCP server configuration JSON and exit (useful for manual runner setup)",
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


def load_prd(path: str) -> List[Dict]:
    """Load and parse the PRD JSON file."""
    with open(path, "r") as f:
        prd = json.load(f)

    tasks = prd.get("tasks", prd.get("features", prd.get("items", [])))

    if not isinstance(tasks, list):
        raise ValueError("PRD must contain a 'tasks', 'features', or 'items' list")

    return tasks


def filter_incomplete_tasks(tasks: List[Dict]) -> List[Dict]:
    """Filter tasks to only include incomplete ones."""
    incomplete = []

    for task in tasks:
        if not isinstance(task, dict):
            continue

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

        if not is_passing:
            incomplete.append(task)

    return incomplete


def get_prd_stats(path: str) -> Tuple[int, int, int]:
    """Read prd.json and count passing/not passing tasks."""
    if not os.path.exists(path):
        return (0, 0, 0)

    try:
        tasks = load_prd(path)

        passing = 0
        not_passing = 0

        for task in tasks:
            if isinstance(task, dict):
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
    """Read prd.json and group tasks by category."""
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

                cat_passing, total = categories[category]
                categories[category] = (
                    cat_passing + (1 if is_passing else 0),
                    total + 1,
                )

        return categories
    except (json.JSONDecodeError, IOError, ValueError):
        return {}


def all_tasks_complete(path: str) -> bool:
    """Check if all tasks in the PRD are complete."""
    passing, not_passing, total = get_prd_stats(path)
    return total > 0 and not_passing == 0


def validate_paths(prd_path: str) -> None:
    """Validate that paths are correct."""
    if not os.path.exists(prd_path):
        raise FileNotFoundError(f"PRD file not found: {prd_path}")

    try:
        load_prd(prd_path)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in PRD file: {e}")
    except ValueError as e:
        raise ValueError(f"Invalid PRD structure: {e}")


def build_system_prompt() -> str:
    """Build the system prompt with MCP tool instructions."""
    prompt = """You are working on implementing tasks from a Product Requirements Document (PRD).

MCP TOOLS AVAILABLE:
You have access to MCP tools from the "ralph-tasks" server for managing task workflow:

1. list_tasks()
   - Lists all incomplete tasks from the PRD
   - Use this to see what work needs to be done

2. get_task(task_id)
   - Get detailed information about a specific task
   - Use this to understand task requirements

3. start_task(task_id)
   - Mark a task as in-progress
   - Call this when you begin working on a task

4. complete_task(task_id, notes)
   - Mark a task as completed/passing
   - Call this after you have successfully implemented and tested the task
   - The 'notes' parameter is optional

5. fail_task(task_id, reason)
   - Mark a task as failed with a reason
   - Call this when you encounter blockers or cannot complete the task

6. get_progress()
   - Get current session progress statistics
   - Shows total tasks, completed, remaining, etc.

WORKFLOW:
1. Call list_tasks() to see the incomplete tasks
2. Select the highest priority task
3. Call start_task(task_id) to mark it in-progress
4. Work on implementing that task
5. Call complete_task(task_id) to mark the task as done
6. If you cannot complete it, call fail_task(task_id, reason)

Begin by calling list_tasks() and selecting a task to work on."""

    return prompt


# ============================================================================
# Iteration Runner
# ============================================================================


def parse_task_log(
    log_path: str, after_timestamp: Optional[str] = None
) -> Tuple[List[Dict], Optional[str]]:
    """Parse the task completion log file for new events after a given timestamp.

    Returns a tuple of (events, latest_timestamp) where events is a list of dicts
    with keys: 'type' (start/complete/fail), 'task_id', 'timestamp'.
    """
    events = []
    latest_timestamp = after_timestamp

    if not os.path.exists(log_path):
        return (events, latest_timestamp)

    try:
        with open(log_path, "r") as f:
            # Parse XML-like tags: <start>task-id</start>, <complete>task-id</complete>, <fail>task-id</fail>
            for line in f:
                # Extract timestamp
                timestamp_match = re.search(r"<time>(.*?)</time>", line)
                timestamp = timestamp_match.group(1) if timestamp_match else None

                # Skip entries at or before the last seen timestamp
                if after_timestamp and timestamp and timestamp <= after_timestamp:
                    continue

                # Track the latest timestamp
                if timestamp and (
                    latest_timestamp is None or timestamp > latest_timestamp
                ):
                    latest_timestamp = timestamp

                # Check for start event
                start_match = re.search(r"<start>(.*?)</start>", line)
                if start_match:
                    events.append(
                        {
                            "type": "start",
                            "task_id": start_match.group(1),
                            "timestamp": timestamp,
                        }
                    )
                    continue

                # Check for complete event
                complete_match = re.search(r"<complete>(.*?)</complete>", line)
                if complete_match:
                    events.append(
                        {
                            "type": "complete",
                            "task_id": complete_match.group(1),
                            "timestamp": timestamp,
                        }
                    )
                    continue

                # Check for fail event
                fail_match = re.search(r"<fail>(.*?)</fail>", line)
                if fail_match:
                    events.append(
                        {
                            "type": "fail",
                            "task_id": fail_match.group(1),
                            "timestamp": timestamp,
                        }
                    )
                    continue

        return (events, latest_timestamp)
    except Exception as e:
        print(f"[Ralph] Warning: Error parsing task log: {e}", file=sys.stderr)
        return (events, latest_timestamp)


def run_iteration(
    runner: str,
    prompt: str,
    # mcp_config: str,
    timeout: Optional[int],
    detector: RepetitionDetector,
    prd_path: str,
    log_path: str,
    last_timestamp: str,
    verbose: bool = False,
) -> Tuple[IterationResult, Optional[str]]:
    """Run a single iteration of the AI agent."""
    global current_process

    # Build command from template
    cmd_template = RUNNER_TEMPLATES[runner]
    cmd_list = [
        part.format(prompt=prompt) if "{" in part else part for part in cmd_template
    ]

    # Join command parts into a string for shell execution
    cmd_parts = []
    for part in cmd_list:
        if " " in part or "\n" in part:
            escaped = part.replace('"', '\\"')
            cmd_parts.append(f'"{escaped}"')
        else:
            cmd_parts.append(part)

    cmd = " ".join(cmd_parts)
    cmd = cmd.replace("\n", " \\n ")

    # if verbose:
    #     print(f"\n[Ralph] Running command: {cmd[:400]}...", file=sys.stderr)

    try:
        import platform

        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
            "shell": True,
        }

        if platform.system() == "Windows":
            popen_kwargs["creationflags"] = 0x00000200  # CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        process = subprocess.Popen(cmd, **popen_kwargs)
        current_process = process

        start_time = time.time()

        # Track timestamp to detect new events
        current_timestamp = last_timestamp

        # Shared state for the output reading thread
        stop_reason: List[Optional[str]] = [None]
        task_id_detected: List[Optional[str]] = [None]
        thread_done = threading.Event()

        def read_output_worker():
            """Background thread that reads and processes subprocess output."""
            current_line = ""

            try:
                while True:
                    if process.poll() is not None:
                        remaining = process.stdout.read()
                        if remaining:
                            sys.stdout.write(remaining)
                            sys.stdout.flush()
                        break

                    if platform.system() == "Windows":
                        try:
                            char = process.stdout.read(1)
                            if not char:
                                continue

                            sys.stdout.write(char)
                            sys.stdout.flush()
                            current_line += char

                            if char == "\n":
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
                        import select

                        ready, _, _ = select.select([process.stdout], [], [], 0.1)
                        if ready:
                            char = process.stdout.read(1)
                            if not char:
                                break

                            sys.stdout.write(char)
                            sys.stdout.flush()
                            current_line += char

                            if char == "\n":
                                if detector.add_line(current_line):
                                    stop_reason[0] = "repetition"
                                    process.terminate()
                                    break
                                current_line = ""
            finally:
                thread_done.set()

        output_thread = threading.Thread(target=read_output_worker, daemon=True)
        output_thread.start()

        # Main thread monitors for timeout and task log changes
        check_interval = 2.0  # Check log every 2 seconds
        last_check = time.time()

        while not thread_done.is_set():
            time.sleep(0.1)

            # Check for timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                process.terminate()
                thread_done.wait(timeout=5)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                current_process = None
                return (
                    IterationResult(stopped_early=True, reason="timeout"),
                    current_timestamp,
                )

            # Check if output thread detected a stop condition
            if stop_reason[0] is not None:
                thread_done.wait(timeout=5)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                current_process = None
                return (
                    IterationResult(
                        stopped_early=True,
                        reason=stop_reason[0],
                        task_id=task_id_detected[0],
                    ),
                    current_timestamp,
                )

            # Periodically check task log for completion/failure events
            if time.time() - last_check >= check_interval:
                last_check = time.time()
                events, new_timestamp = parse_task_log(log_path, current_timestamp)

                if new_timestamp:
                    current_timestamp = new_timestamp

                for event in events:
                    if event["type"] == "complete":
                        if verbose:
                            print(
                                f"\n[Ralph] Task completed via MCP: {event['task_id']}",
                                file=sys.stderr,
                            )
                        # End the current iteration immediately
                        process.terminate()
                        thread_done.wait(timeout=5)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        current_process = None
                        return (
                            IterationResult(
                                completed=True,
                                reason="task_complete",
                                task_id=event["task_id"],
                            ),
                            current_timestamp,
                        )
                    elif event["type"] == "fail":
                        if verbose:
                            print(
                                f"\n[Ralph] Task failed via MCP: {event['task_id']}",
                                file=sys.stderr,
                            )
                        # End the current iteration immediately
                        process.terminate()
                        thread_done.wait(timeout=5)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        current_process = None
                        return (
                            IterationResult(
                                completed=True,
                                reason="task_failed",
                                task_id=event["task_id"],
                            ),
                            current_timestamp,
                        )
                    elif event["type"] == "start" and verbose:
                        print(
                            f"\n[Ralph] Task started via MCP: {event['task_id']}",
                            file=sys.stderr,
                        )

        output_thread.join(timeout=5)
        exit_code = process.wait()
        current_process = None

        return (IterationResult(completed=True, exit_code=exit_code), current_timestamp)

    except FileNotFoundError:
        current_process = None
        print(f"\nError: Runner '{runner}' not found in PATH", file=sys.stderr)
        return (
            IterationResult(completed=False, stopped_early=True, reason="error"),
            last_timestamp,
        )
    except Exception as e:
        current_process = None
        print(f"\nError running iteration: {e}", file=sys.stderr)
        return (
            IterationResult(completed=False, stopped_early=True, reason="error"),
            last_timestamp,
        )


# ============================================================================
# Status Display and Utilities
# ============================================================================


def get_commits_since(timestamp: float) -> int:
    """Count git commits made since the given Unix timestamp."""
    try:
        since_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

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

    passing, not_passing, total = get_prd_stats(prd_path)
    categories = get_category_breakdown(prd_path)
    commits = get_commits_since(session_start_time)

    max_iter_str = str(max_iterations) if max_iterations else "∞"
    print(f"\n{'=' * 80}")
    print(f"  ITERATION {iteration}/{max_iter_str}")
    print(f"{'=' * 80}")

    time_info = f"  🕐 Current Time: {current_time}"
    time_info += f"  │  Session: {format_duration(session_elapsed)}"
    if prev_iteration_time is not None:
        time_info += f"  │  Prev Iteration: {format_duration(prev_iteration_time)}"
    print(time_info)

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

    if current_task:
        print(f"  🎯 Current Task: {current_task}")

    if categories:
        cat_parts = []
        for cat, (cat_pass, cat_total) in sorted(categories.items()):
            marker = "✓" if cat_pass == cat_total else ""
            cat_parts.append(f"{cat}: {cat_pass}/{cat_total}{marker}")
        print(f"  📁 Categories: {' | '.join(cat_parts)}")

    completion_rate = 0
    if completed_iterations > 0:
        completion_rate = (
            stop_reasons.get("task_complete", 0) / completed_iterations
        ) * 100
    session_info = f"  📈 Session: {completed_iterations} iterations"
    session_info += f" │ Commits: {commits}"
    session_info += f" │ Rate: {completion_rate:.0f}% complete"
    print(session_info)

    if completed_iterations > 0:
        stop_parts = []
        for reason in [
            "task_complete",
            "task_failed",
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
    """Kill a process and all its children."""
    import platform

    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=5,
            )
        else:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        pass


def handle_sigint(signum, frame):
    """Handle SIGINT (CTRL+C) signals."""
    global exit_requested, current_process, sigint_count

    sigint_count += 1

    if sigint_count == 1:
        exit_requested = True
        print("\n[Ralph] Exit requested. Will exit after current iteration completes.")
        print("[Ralph] Press CTRL+C again to exit immediately.")
    else:
        print("\n[Ralph] Force exit requested. Terminating immediately...")
        if current_process is not None:
            try:
                kill_process_tree(current_process.pid)
            except Exception:
                pass
        sys.exit(130)


# ============================================================================
# Main Loop
# ============================================================================


def get_mcp_config_dict(prd_path: str) -> dict:
    """Get the MCP configuration as a dictionary (without writing to file)."""
    script_dir = Path(__file__).parent.resolve()
    server_path = script_dir / "ralph_server" / "server.py"
    prd_abs = Path(prd_path).resolve()
    log_file = Path.cwd() / "ralph_server.log"

    # Use 'uv run' to ensure the correct Python environment with mcp package
    return {
        "mcpServers": {
            "ralph-tasks": {
                "type": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "--directory",
                    str(script_dir),
                    "python",
                    str(server_path),
                    "--prd-path",
                    str(prd_abs),
                    "--log-file",
                    str(log_file),
                ],
            }
        }
    }


def main():
    """Main entry point."""
    global exit_requested

    args = parse_args()

    # Handle --print-mcp-config flag
    if args.print_mcp_config:
        # Only need to check if PRD exists for config generation
        prd_path = Path(args.prd_path)
        if not prd_path.exists():
            print(f"Error: PRD file not found: {args.prd_path}", file=sys.stderr)
            sys.exit(1)
        config = get_mcp_config_dict(args.prd_path)
        print(json.dumps(config, indent=2))
        sys.exit(0)

    # Validate paths
    try:
        validate_paths(args.prd_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_ascii_art()

    # Register signal handler
    signal.signal(signal.SIGINT, handle_sigint)

    # Create MCP config for the runner
    # mcp_config = create_mcp_config(args.prd_path)

    # Print configuration
    print(f"{'=' * 60}")
    print("Ralph MCP Configuration")
    print(f"{'=' * 60}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"PRD Path:          {args.prd_path}")
    print(f"Runner:            {args.runner}")
    # print(f"MCP Config:        {mcp_config}")
    print(
        f"Max Iterations:    {args.max_iterations if args.max_iterations else 'unlimited'}"
    )
    print(f"Timeout:           {args.timeout if args.timeout else 'none'}")
    print(f"Rep. Threshold:    {args.repetition_threshold}")
    print(f"History Size:      {args.history_size}")
    print(f"Verbose:           {args.verbose}")
    print(f"{'=' * 60}\n")

    # Initialize
    detector = RepetitionDetector(
        history_size=args.history_size, threshold=args.repetition_threshold
    )

    # Determine task completion log path
    task_log_path = str(Path(__file__).parent.resolve() / "ralph_task_completions.log")

    # Clear old data from the log
    os.unlink(task_log_path)
    Path(task_log_path).write_text("")

    session_start_time = time.time()
    prev_iteration_time: Optional[float] = None
    completed_iterations = 0
    stop_reasons: Dict[str, int] = {
        "task_complete": 0,
        "task_failed": 0,
        "timeout": 0,
        "repetition": 0,
        "error": 0,
        "normal": 0,
    }
    current_task: Optional[str] = None
    last_log_timestamp = datetime.now().isoformat()

    # Main loop
    iteration = 0
    while not all_tasks_complete(args.prd_path):
        iteration += 1

        if exit_requested:
            print("\n[Ralph] Exiting gracefully - not starting next iteration")
            break

        if args.max_iterations and iteration > args.max_iterations:
            print(f"\n[Ralph] Reached maximum iterations ({args.max_iterations})")
            break

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

        try:
            tasks = load_prd(args.prd_path)
            incomplete = filter_incomplete_tasks(tasks)

            if not incomplete:
                print("\n[Ralph] All tasks complete!")
                break

            if args.verbose:
                print(
                    f"[Ralph] Found {len(incomplete)} incomplete tasks",
                    file=sys.stderr,
                )

        except Exception as e:
            print(f"\n[Ralph] Error loading PRD: {e}", file=sys.stderr)
            break

        # Write filtered PRD to working directory (for reference)
        # try:
        #     write_filtered_prd(incomplete)
        # except Exception as e:
        #     print(f"\n[Ralph] Error writing filtered PRD: {e}", file=sys.stderr)
        #     break

        # Build prompt
        prompt = build_system_prompt()

        # Run iteration
        iteration_start_time = time.time()

        result, last_log_timestamp = run_iteration(
            runner=args.runner,
            prompt=prompt,
            # mcp_config=mcp_config,
            timeout=args.timeout,
            detector=detector,
            prd_path=args.prd_path,
            log_path=task_log_path,
            last_timestamp=last_log_timestamp,
            verbose=args.verbose,
        )

        prev_iteration_time = time.time() - iteration_start_time
        completed_iterations += 1

        # Handle results
        if result.stopped_early:
            if result.reason == "repetition":
                stop_reasons["repetition"] += 1
                print("\n[Ralph] Repetition detected, moving to next iteration\n")
            elif result.reason == "timeout":
                stop_reasons["timeout"] += 1
                print("\n[Ralph] Timeout reached, moving to next iteration\n")
            elif result.reason == "error":
                stop_reasons["error"] += 1
                print("\n[Ralph] Error occurred, stopping execution\n")
                sys.exit(1)
        elif result.reason == "task_complete":
            stop_reasons["task_complete"] += 1
            current_task = result.task_id
            print("\n[Ralph] Task completed successfully, moving to next iteration\n")
        elif result.reason == "task_failed":
            stop_reasons["task_failed"] += 1
            current_task = result.task_id
            print("\n[Ralph] Task failed, moving to next iteration\n")
        else:
            stop_reasons["normal"] += 1

        detector.reset()

    # Cleanup temp config
    # try:
    #     Path(mcp_config).unlink(missing_ok=True)
    # except Exception:
    #     pass

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
