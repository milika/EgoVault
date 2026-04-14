# Contributing to EgoVault

Thank you for your interest in contributing! This document explains how to
report bugs, suggest features, and submit code changes.

---

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
By participating you agree to abide by its terms.

---

## Reporting bugs

1. Search [existing issues](https://github.com/milika/EgoVault/issues) to
   avoid duplicates.
2. Open a new issue and fill in the bug-report template.
3. Include:
   - EgoVault version (`pip show egovault`)
   - Python version and OS
   - Minimal steps to reproduce
   - Expected vs. actual behaviour
   - Relevant log output (set `log_level = "DEBUG"` in `egovault.toml`)

---

## Suggesting features

Open an issue with the **Feature request** template and describe:
- The problem you are solving
- Your proposed solution
- Alternatives you considered

---

## Submitting a pull request

### 1. Fork and clone

```bash
git clone https://github.com/milika/EgoVault.git
cd egovault
```

### 2. Set up the dev environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev,local,web,mcp,reranker]"
```

### 3. Make your changes

- Follow the existing code style (PEP 8, `ruff` for linting).
- Keep changes focused — one logical change per PR.
- Add or update tests under `tests/` for any new behaviour.

### 4. Update the changelog

Add an entry to `CHANGELOG.md` under the `[Unreleased]` section using the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format:

```markdown
### Added / Changed / Fixed / Removed
- Short description of the change.
```

Only omit a changelog entry for trivial fixes (typos, comment changes, test
fixture tweaks that carry no user-visible effect).

### 5. Run the tests

```bash
pytest
```

All tests must pass before a PR will be reviewed.

### 6. Open a pull request

- Target the `main` branch.
- Write a clear PR description: what changed and why.
- Reference any related issues (`Closes #123`).
- Confirm your `CHANGELOG.md` entry is present.

---

## Development tips

| Command | Purpose |
|---------|---------|
| `pytest -x` | Stop on first failure |
| `pytest --cov=egovault` | Coverage report |
| `ruff check .` | Lint the codebase |
| `ego chat` | Manual smoke-test of the RAG pipeline |
| `ego web` | Manual smoke-test of the Streamlit browser UI |

---

## Project structure

```
egovault/
  adapters/     # Data-source adapters (Gmail, local inbox, …)
  agent/        # Agent loop, tool definitions, slash commands
  core/         # Schema, store, registry
  frontends/    # Terminal REPL, Streamlit web UI, Telegram bot
  processing/   # RAG pipeline, enrichment
  output/       # Markdown / CSV exporters
  utils/        # Chunking, embeddings, hashing, LLM helpers
tests/          # pytest test suite
docs/           # How it works, installation guide
```

---

Questions? Open an issue or start a discussion on GitHub.
