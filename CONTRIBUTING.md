# Contributing to TQNN

Thank you for your interest in contributing to the Topological Quantum Neural Networks project.

## Development Setup

```bash
git clone https://github.com/IsolatedSingularity/Topological-Quantum-Neural-Networks.git
cd Topological-Quantum-Neural-Networks
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -e .
```

## Running Tests

```bash
pytest tests/ -v
```

On headless Linux (CI), prefix with `xvfb-run` since some tests import tkinter:

```bash
xvfb-run pytest tests/ -v
```

## Project Structure

```
tqnn/                   Main package
  simulation/           Real-time tensor network simulator GUI (tkinter)
  classifier/           Interactive TQNN pattern classifier GUI
  cobordism/            Cobordism evolution viewer GUI
  visualization/        Static and animated plot generators
  helpers.py            TQNNPerceptron, spin-network encoding, noise injection
tests/                  pytest suite
Plots/                  Generated figures, GIFs, and screenshots
```

## Code Style

- Python 3.10+ syntax
- **camelCase** for all identifiers (overrides PEP 8 snake_case)
- Type hints on public APIs
- Docstrings for classes and public methods
- Consistent dark theme (`#1a1a1a` background, `#00ff88` accent) for all GUIs and plots

## Linting and Type Checking

Before submitting, ensure your changes pass both linters:

```bash
ruff check tqnn/
mypy tqnn/ --ignore-missing-imports
```

Both tools are configured in `pyproject.toml`. Install dev dependencies with:

```bash
pip install -e ".[dev]"
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Make your changes and add tests if appropriate
4. Run `pytest tests/ -v` to ensure all tests pass
5. Commit with a clear message
6. Open a pull request

## Reporting Issues

Open a GitHub issue with:
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
