# Ralph - Advanced AI Agent Runner

An advanced Python-based AI agent runner that executes AI workflows with iteration control, timeouts, and intelligent repetition detection.

## Features

- **Multiple Runner Support**: Works with `crush`, `opencode`, and `claude-code`
- **Iteration Control**: Run agents multiple times with configurable iteration count
- **Smart Timeout**: Hard timeout that terminates runaway processes
- **Repetition Detection**: Automatically detects and stops repeated output patterns
- **Promise Detection**: Recognizes completion markers (`<promise>COMPLETE</promise>`)
- **Real-time Streaming**: All output streams to stdout as it happens

## About Ralph Wiggum

This project is inspired by Geoffrey Huntley's ["Ralph Wiggum" pattern](https://ghuntley.com/ralph/) - a novel approach to AI-driven development where an AI agent works autonomously through a Product Requirements Document (PRD), implementing one feature at a time until completion.

This Python implementation extends the original concept with advanced features including multiple runner support, intelligent repetition detection, configurable timeouts, and sophisticated iteration control. The name "Ralph" pays homage to the Simpsons character who famously said "I'm helping!" - capturing the spirit of an AI assistant that persistently works through tasks, learning and improving with each iteration.

## Installation

Requires Python 3.13 or higher. No external dependencies needed.

```bash
python ralph.py --help
```

## Quick Start

Get started with Ralph in three simple steps:

### 1. Create a minimal prd.json

```json
{
  "product": "Hello World App",
  "goal": "Create a simple Python application",
  "principles": ["Keep it simple", "Make it work"],
  "tasks": [
    {
      "id": "create-hello-001",
      "category": "feature",
      "description": "Create hello.py that prints Hello, World!",
      "steps": [
        "Create hello.py file",
        "Add print statement",
        "Test by running python hello.py"
      ],
      "passes": false
    }
  ]
}
```

### 2. Run Ralph

```bash
python ralph.py --iterations 1
```

### 3. Check the results

Ralph will:
- Read your prd.json
- Implement the feature
- Update prd.json marking the task as `passes: true`
- Make a git commit
- Output `<promise>COMPLETE</promise>` when done

For more complex projects, see the [Understanding PRD.json](#understanding-prdjson) section below.

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

Ralph orchestrates an AI agent to work through your PRD autonomously, with intelligent safeguards to prevent runaway processes.

### The Big Picture

```
Your PRD (prd.json)
        ↓
    Ralph reads tasks
        ↓
    Identifies incomplete work (passes: false)
        ↓
    Launches AI runner (crush/opencode/claude-code)
        ↓
    AI implements feature
        ↓
    AI updates prd.json (passes: true)
        ↓
    AI makes git commit
        ↓
    AI outputs <promise>COMPLETE</promise>
        ↓
    Next iteration begins
```

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

## Understanding PRD.json

### What is PRD.json?

PRD stands for **Product Requirements Document**. In the Ralph workflow, your prd.json file serves as:

1. **Task List**: All features and work items to be completed
2. **State Tracker**: Each task has a `passes` field indicating completion status
3. **AI Instructions**: The AI reads this file to decide what to work on next
4. **Progress Record**: Updated automatically by the AI as work completes

### PRD.json Structure

```json
{
  "product": "Your Product Name",
  "goal": "High-level description of what you're building",
  "principles": [
    "Core principle 1 (e.g., 'Keep it simple')",
    "Core principle 2 (e.g., 'Test everything')",
    "Core principle 3"
  ],
  "tasks": [
    {
      "id": "unique-task-id-001",
      "category": "feature",  // feature, bugfix, refactor, test, docs
      "description": "Human-readable description of what needs to be done",
      "steps": [
        "Concrete step 1",
        "Concrete step 2",
        "Verification step (e.g., 'Run tests and confirm they pass')"
      ],
      "passes": false  // AI changes this to true when complete
    }
  ]
}
```

### How Ralph Uses PRD.json

**The Workflow Loop:**

```
┌─────────────────────────────────────────────────────┐
│  1. Ralph reads prd.json                            │
│  2. AI identifies highest-priority incomplete task  │
│     (where passes: false)                           │
│  3. AI implements the feature                       │
│  4. AI updates prd.json (passes: false → true)      │
│  5. AI makes git commit                             │
│  6. AI outputs <promise>COMPLETE</promise>          │
│  7. Next iteration begins (back to step 1)          │
└─────────────────────────────────────────────────────┘
```

### Example: Task Progression

**Before implementation:**
```json
{
  "id": "add-login-001",
  "category": "feature",
  "description": "Add user login endpoint",
  "steps": [
    "Create /login POST endpoint",
    "Validate username and password",
    "Return JWT token on success"
  ],
  "passes": false
}
```

**After implementation:**
```json
{
  "id": "add-login-001",
  "category": "feature",
  "description": "Add user login endpoint",
  "steps": [
    "Create /login POST endpoint",
    "Validate username and password",
    "Return JWT token on success"
  ],
  "passes": true  // ← AI updated this
}
```

### Creating Your PRD.json

Follow this four-step workflow to create effective PRD files:

#### Step 1: Create a Plan in Claude Code

Use Claude Code to plan your project and break it down into features:

```bash
claude "I want to build a blog engine with authentication.
Help me break this down into implementable features."
```

#### Step 2: Translate to prd.json Format

Take Claude's plan and use another AI tool (ChatGPT, Claude web, etc.) to convert it:

**Prompt to use:**
```
Convert this plan into prd.json format with:
- product: Brief product name
- goal: One-sentence description
- principles: 3-5 key principles to follow
- tasks: Array of tasks, each with id, category, description, steps, and passes: false
```

#### Step 3: Refine the Items

Review each task and ensure:

- **Atomic**: Each task has a single, clear responsibility
- **Simple**: Can be completed in one AI session (typically under 10 minutes)
- **Testable**: Steps include verification (e.g., "Run tests and confirm they pass")
- **Independent**: Tasks can be done in any order, or dependencies are clear
- **Specific**: Enough detail to guide implementation, but not overly prescriptive

**Task Categories:**
- `feature`: New functionality
- `bugfix`: Fix broken behavior
- `refactor`: Improve code structure
- `test`: Add test coverage
- `docs`: Documentation updates

**Good vs. Bad Tasks:**

❌ **Too vague:**
```json
{"description": "Make the app better"}
```

✅ **Clear and specific:**
```json
{
  "description": "Add pagination to blog post list endpoint",
  "steps": [
    "Update /posts endpoint to accept page and limit params",
    "Return paginated results with total count",
    "Add tests for pagination edge cases"
  ]
}
```

#### Step 4: Run ralph.py

```bash
python ralph.py --iterations 10 --timeout 600
```

Ralph will work through your PRD, implementing one feature at a time.

### Tips for Effective PRDs

1. **Start small**: Begin with 3-5 tasks to validate your workflow
2. **Be specific in steps**: Help the AI understand exactly what "done" looks like
3. **Include verification**: Always add a step like "Run tests" or "Manually test feature"
4. **Order matters**: Place foundational tasks first (e.g., database setup before endpoints)
5. **Use progress.txt**: The AI appends notes here between iterations

## Examples

### Complete Workflow Example

Here's a real-world example showing Ralph in action:

**Initial prd.json:**
```json
{
  "product": "Simple Calculator API",
  "goal": "Build a REST API for basic math operations",
  "principles": ["Keep it simple", "Test everything"],
  "tasks": [
    {
      "id": "setup-flask-001",
      "category": "feature",
      "description": "Setup Flask application with basic structure",
      "steps": [
        "Create app.py with Flask initialization",
        "Add health check endpoint",
        "Test with curl"
      ],
      "passes": false
    },
    {
      "id": "add-endpoint-001",
      "category": "feature",
      "description": "Add POST /calculate endpoint",
      "steps": [
        "Create /calculate endpoint accepting operation and numbers",
        "Support add, subtract, multiply, divide",
        "Return JSON with result"
      ],
      "passes": false
    },
    {
      "id": "add-tests-001",
      "category": "test",
      "description": "Add unit tests for calculator",
      "steps": [
        "Create test_calculator.py",
        "Test all operations",
        "Test error cases"
      ],
      "passes": false
    }
  ]
}
```

**Run Ralph:**
```bash
python ralph.py --iterations 3 --timeout 300
```

**What Happens:**

**Iteration 1:**
- Ralph launches the AI runner
- AI reads prd.json, picks "setup-flask-001"
- AI creates app.py with Flask setup
- AI updates prd.json (`"passes": true` for setup-flask-001)
- AI commits: "Setup Flask application with basic structure"
- AI outputs `<promise>COMPLETE</promise>`

**Iteration 2:**
- AI reads updated prd.json
- AI picks "add-endpoint-001" (next incomplete task)
- AI implements /calculate endpoint
- AI updates prd.json (`"passes": true` for add-endpoint-001)
- AI commits: "Add POST /calculate endpoint"
- AI outputs `<promise>COMPLETE</promise>`

**Iteration 3:**
- AI reads updated prd.json
- AI picks "add-tests-001"
- AI creates test_calculator.py
- AI updates prd.json (`"passes": true` for add-tests-001)
- AI commits: "Add unit tests for calculator"
- AI outputs `<promise>COMPLETE</promise>`

**Final prd.json:**
```json
{
  "product": "Simple Calculator API",
  "goal": "Build a REST API for basic math operations",
  "principles": ["Keep it simple", "Test everything"],
  "tasks": [
    {
      "id": "setup-flask-001",
      "passes": true  // ← Completed in iteration 1
    },
    {
      "id": "add-endpoint-001",
      "passes": true  // ← Completed in iteration 2
    },
    {
      "id": "add-tests-001",
      "passes": true  // ← Completed in iteration 3
    }
  ]
}
```

**Git history:**
```
* abc1234 - Add unit tests for calculator
* def5678 - Add POST /calculate endpoint
* ghi9012 - Setup Flask application with basic structure
```

**progress.txt contents:**
```
[Iteration 1] Setup Flask app with health check at /health. Ready for endpoints.
[Iteration 2] Added /calculate endpoint with support for +, -, *, /. Handles errors.
[Iteration 3] Full test coverage added. All tests passing.
```

### More Examples

**Full PRD Workflow** - Run 10 iterations with 10-minute timeout per iteration:
```bash
python ralph.py --iterations 10 --timeout 600
```

**Quick Test Run** - Test with a simple prompt, 2 iterations:
```bash
python ralph.py --prompt "List all TODO comments in the code" --iterations 2
```

**Aggressive Repetition Detection** - Stop faster when repetition is detected:
```bash
python ralph.py --repetition-threshold 2 --check-interval 15
```

**Using Different Runners:**
```bash
# Use opencode instead of crush
python ralph.py --runner opencode --iterations 5

# Use claude-code
python ralph.py --runner claude-code --iterations 3
```

## Original Script

This is an advanced Python reimplementation of [ralph-once.sh](ralph-once.sh), adding:

- Multiple runner support
- Iteration control
- Timeout management
- Repetition detection
- Better error handling
- Configurable everything

## Troubleshooting

### Repetition Detection Triggering Too Early

**Problem**: Ralph stops iterations because it thinks the AI is repeating itself, but work is still being done.

**Solution**: Adjust the repetition detection parameters:
```bash
# Increase the threshold (more consecutive repeats needed)
python ralph.py --repetition-threshold 5

# Increase the check interval (larger buffer windows)
python ralph.py --check-interval 50

# Or both
python ralph.py --repetition-threshold 5 --check-interval 50
```

### AI Agent Gets Stuck

**Problem**: The AI enters an infinite loop or keeps doing the same thing.

**Solution**: This is exactly what timeouts and repetition detection are for:
```bash
# Add a timeout to force termination
python ralph.py --timeout 300 --iterations 5

# Use aggressive repetition detection
python ralph.py --repetition-threshold 2 --check-interval 15
```

### Tasks Not Being Completed

**Problem**: The AI works on a task but doesn't mark it as `passes: true` in prd.json.

**Solutions**:
1. Make task steps more explicit about updating prd.json
2. Check if the AI encountered errors (look at console output)
3. Verify the task is achievable in one iteration
4. Check progress.txt for AI's notes on what went wrong

**Better task example:**
```json
{
  "steps": [
    "Implement the feature",
    "Test that it works",
    "Update this task in prd.json to passes: true",
    "Commit with git"
  ]
}
```

### Manual Intervention Needed

**Problem**: You need to manually fix something between iterations.

**Solution**: Ralph works with standard git, so you can:
1. Stop Ralph (Ctrl+C)
2. Make your manual changes
3. Commit your changes: `git commit -am "Manual fix"`
4. Restart Ralph: `python ralph.py --iterations N`

Ralph will pick up where it left off by reading the current prd.json state.

### Wrong Task Priority

**Problem**: The AI picks tasks in the wrong order.

**Solutions**:
1. **Reorder tasks in prd.json**: Place higher-priority tasks first
2. **Use dependencies in task descriptions**: "After completing setup-001, implement..."
3. **Be explicit in principles**: Add "Complete tasks in order" to the principles array

### No Progress After Multiple Iterations

**Problem**: Ralph runs but nothing changes.

**Debug steps**:
1. Check if the runner (crush/opencode/claude-code) is installed and working
2. Verify prd.json is valid JSON (use `python -m json.tool prd.json`)
3. Look at console output for error messages
4. Try with `--prompt` override to test the runner: `python ralph.py --prompt "echo hello" --iterations 1`
5. Check that git is configured (name, email)

### Promise Markers Not Working

**Problem**: AI outputs `<promise>COMPLETE</promise>` but Ralph doesn't detect it.

**Solution**: This is usually correct behavior! The promise marker:
- Stops the current iteration early
- Moves to the next iteration
- Does NOT stop all iterations

If all work is truly done, the AI should stop making changes and the remaining iterations will complete with no work done.

## License

MIT License