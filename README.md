# Ralph - Advanced AI Agent Runner

An advanced Python-based AI agent runner that executes AI workflows with iteration control, timeouts, and intelligent repetition detection.

## Features

- **Multiple Runner Support**: Works with `crush`, `opencode`, and `claude-code`
- **Iteration Control**: Run agents multiple times with configurable iteration count
- **Smart Timeout**: Hard timeout that terminates runaway processes
- **Repetition Detection**: Automatically detects and stops repeated output patterns
- **Promise Detection**: Recognizes completion markers (`<promise>COMPLETE</promise>`)
- **Real-time Streaming**: All output streams to stdout as it happens

## Installation

Requires Python 3.13 or higher. No external dependencies needed.

```bash
python ralph.py --help
```

## Usage

### Basic Usage

Run once with default settings (crush runner, default PRD prompt):

```bash
python ralph.py
```

### Multiple Iterations

Run the agent 5 times:

```bash
python ralph.py --iterations 5
```

### With Timeout

Run with a 5-minute (300 second) timeout per iteration:

```bash
python ralph.py --iterations 3 --timeout 300
```

### Different Runner

Use opencode instead of crush:

```bash
python ralph.py --runner opencode --iterations 2
```

### Custom Prompt

Override the default PRD-based prompt:

```bash
python ralph.py --prompt "Fix the authentication bug in main.py"
```

### Custom File References

Change which files are referenced in the default prompt:

```bash
python ralph.py --file-refs "@requirements.json @status.txt"
```

### Advanced Configuration

Fine-tune repetition detection:

```bash
python ralph.py \
  --iterations 10 \
  --timeout 600 \
  --repetition-threshold 5 \
  --check-interval 50
```

## Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--runner` | `-r` | Runner to use (crush, opencode, claude-code) | crush |
| `--iterations` | `-i` | Number of iterations to run | 1 |
| `--timeout` | `-t` | Timeout in seconds for each iteration | none |
| `--repetition-threshold` | `-rt` | Consecutive repeats before stopping | 3 |
| `--check-interval` | `-ci` | Lines to buffer before checking repetition | 25 |
| `--file-refs` | `-f` | File references in default prompt | @prd.json @progress.txt |
| `--prompt` | `-p` | Override default prompt entirely | (see below) |

## Default Prompt

When no custom prompt is provided, ralph.py uses this workflow:

```
@prd.json @progress.txt
1. Find the highest-priority feature that is not already passing to work on and work only on that feature.
   This should be the one YOU decide has the highest priority - not necessarily the first in the list.
2. Update the prd.json with the work that was done.
3. Append your progress to the progress.txt file.
   Use this to leave a note for the next person working in the codebase.
4. Make a git commit of that feature using `git commit --all`.
   ONLY WORK ON A SINGLE FEATURE.
   If you are unable to complete a feature or get stuck on a failing test, save your findings in progress.txt and stop.
   If, while implementing the feature, you notice the PRD is complete, output <promise>COMPLETE</promise>.
```

## How It Works

### Iteration Loop

Ralph runs the specified number of iterations, where each iteration:

1. Starts the runner subprocess with the prompt
2. Streams output to stdout in real-time
3. Monitors for:
   - **Promise marker**: `<promise>COMPLETE</promise>` → stops iteration, continues to next
   - **Repetition**: Consecutive repeated patterns → stops iteration, continues to next
   - **Timeout**: Time limit exceeded → kills process, continues to next
4. Resets state between iterations

### Repetition Detection

The repetition detector:

- Buffers output in sliding windows (default: 25 lines)
- Extracts text patterns (sentences/paragraphs longer than 10 chars)
- Compares consecutive windows using Jaccard similarity
- Triggers when similarity > 70% for N consecutive checks (default: 3)
- Stops the current iteration and moves to the next

This prevents agents from getting stuck in output loops.

### Promise Detection

When the runner outputs `<promise>COMPLETE</promise>`, the current iteration stops immediately and the next iteration begins. This allows agents to signal completion while still running remaining iterations.

### Timeout Handling

If an iteration exceeds the timeout:

- The process is terminated gracefully (SIGTERM)
- If it doesn't stop within 5 seconds, it's killed (SIGKILL)
- The next iteration begins
- No error is thrown (timeout is expected behavior)

## Examples

### Full PRD Workflow

Run 10 iterations with 10-minute timeout per iteration:

```bash
python ralph.py --iterations 10 --timeout 600
```

### Quick Test Run

Test with a simple prompt, 2 iterations:

```bash
python ralph.py --prompt "List all TODO comments in the code" --iterations 2
```

### Aggressive Repetition Detection

Stop faster when repetition is detected:

```bash
python ralph.py --repetition-threshold 2 --check-interval 15
```

## Original Script

This is an advanced Python reimplementation of [ralph-once.sh](ralph-once.sh), adding:

- Multiple runner support
- Iteration control
- Timeout management
- Repetition detection
- Better error handling
- Configurable everything

## License

Part of the scripts collection.
