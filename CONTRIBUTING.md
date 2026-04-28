# Contributing

Contributions are welcome. Here's how to get started.

## Setup

```bash
git clone https://github.com/your-username/startup-signal-pipeline.git
cd startup-signal-pipeline
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill in your API keys
```

## Making Changes

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused and descriptive.
3. Ensure your code parses cleanly: `python -m py_compile <file>`.
4. Open a pull request with a clear description of what changed and why.

## Guidelines

- Follow the existing code style (PEP 8, type hints where practical).
- Do not commit `.env`, credential files, or database files.
- Keep new dependencies minimal — add them to `requirements.txt` with pinned versions.
- If you add a new ATS provider, add it to `app/hiring/detect_ats.py` following the existing pattern.

## Reporting Issues

Open a GitHub issue with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behaviour
- Python version and OS
