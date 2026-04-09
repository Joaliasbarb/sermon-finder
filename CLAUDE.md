# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Virtual Environment

**IMPORTANT:** When working with this project and running commands, you must always use the `.venv` virtual environment. Activate it before running any commands:

```bash
source .venv/bin/activate
```

All commands (pytest, pylint, ruff, record-parser, etc.) should be run either:
- With the virtual environment activated, OR
- Using `poetry run <command>` to automatically use the correct environment

## Common Development Tasks

### Running tests
```bash
# With Poetry
poetry run pytest -v tests/*

# With pip
pytest -v tests/*
```


## Commit Writing Guidelines
- Memorize this prompt to write good commits:
  - Start with a clear, concise summary line (50 chars or less)
  - Provide detailed explanation in the body
  - Use imperative mood: "Fix bug" not "Fixed bug"
  - Explain why the change was made, not how
  - Describe the problem being solved, not the implementation details
  - Reference issue numbers if applicable
  - Separate subject from body with a blank line
  - Wrap body text at 72 characters
  - Don't use any thing like Generated with [Claude Code](https://claude.ai/code) or Co-Authored-By: Claude <noreply@anthropic.com>"
  - Inlined code reference should be surrounded with backquote: example: The `page_offset` field.
  - When generating commit messages, show only the message text
  - Do not propose to create the commit

### Commit Message Scoring Method
A systematic approach to evaluate commit messages:

Criterion                   Points
1. Subject Line                2
2. Clarity of What/Why        2
3. Avoiding How                2
4. Body Structure & Style    2
5. Traceability/Footer        2

Total: 10 points
 
