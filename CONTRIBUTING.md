# Contributing to Stock Analysis Multi-Agent System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Quality Standards](#code-quality-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Code Review](#code-review)

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **Git**
- **Google Gemini API Key** ([Get one here](https://makersuite.google.com/app/apikey))

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/single_stock_analysis_agent.git
   cd single_stock_analysis_agent
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies (including dev tools)**
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

5. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

6. **Verify setup**
   ```bash
   # Run linter
   ruff check .

   # Run type checker
   mypy .

   # Run tests
   pytest
   ```

## 🛠️ Development Workflow

### Branch Strategy

- `main` - Production-ready code
- `develop` - Integration branch
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Making Changes

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guide
   - Add tests for new functionality
   - Update documentation as needed

3. **Run quality checks**
   ```bash
   # Lint and auto-fix issues
   ruff check . --fix
   ruff format .

   # Type check
   mypy .

   # Run tests
   pytest --cov=. --cov-report=html
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   **Commit message format:**
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Adding or updating tests
   - `refactor:` Code refactoring
   - `chore:` Maintenance tasks

5. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📏 Code Quality Standards

### Linting & Formatting

We use **Ruff** for both linting and formatting (replaces black, isort, flake8).

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix

# Format code
ruff format .
```

### Type Checking

We use **mypy** for static type checking.

```bash
# Run type checker
mypy .

# Check specific file
mypy path/to/file.py
```

### Code Style Guidelines

- **Line length**: 100 characters
- **Quotes**: Double quotes for strings
- **Imports**: Sorted alphabetically, grouped by standard lib → third-party → local
- **Type hints**: Required for all function signatures
- **Docstrings**: Google-style docstrings for all public functions and classes

**Example:**

```python
"""Module docstring explaining purpose."""

from datetime import datetime
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel

from models.event_profile import EventProfile


def analyze_stock_data(
    symbol: str,
    start_date: datetime,
    end_date: Optional[datetime] = None
) -> List[EventProfile]:
    """
    Analyze stock data for significant events.

    Args:
        symbol: Stock ticker symbol (e.g., 'TSLA')
        start_date: Analysis start date
        end_date: Analysis end date (default: today)

    Returns:
        List of EventProfile objects for significant events

    Raises:
        ValueError: If symbol is invalid or dates are out of range
    """
    # Implementation here
    pass
```

## 🧪 Testing

### Test Structure

```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Multi-component tests
└── conftest.py     # Shared fixtures
```

### Writing Tests

1. **All new code must have tests**
2. **Aim for 80%+ coverage**
3. **Use descriptive test names**
4. **One assertion per test** (when possible)

**Example:**

```python
import pytest

@pytest.mark.unit
def test_config_validation_with_valid_values():
    """Test that config validation passes with valid values."""
    from config import Config

    assert Config.validate() is True


@pytest.mark.unit
def test_event_profile_creation():
    """Test basic EventProfile creation."""
    from models.event_profile import EventProfile
    from datetime import datetime

    profile = EventProfile(
        event_id="TEST_001",
        date=datetime.now(),
        symbol="TSLA",
        open_price=200.0,
        close_price=210.0,
        high_price=212.0,
        low_price=199.0,
        volume=1000000,
        price_change_pct=5.0
    )

    assert profile.symbol == "TSLA"
    assert profile.price_change_pct == 5.0
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Only unit tests
pytest tests/unit/

# Exclude slow tests
pytest -m "not slow"

# Specific test file
pytest tests/unit/test_config.py -v
```

## 🔍 Pull Request Process

### Before Submitting

- [ ] All tests pass (`pytest`)
- [ ] Code is linted (`ruff check .`)
- [ ] Code is formatted (`ruff format .`)
- [ ] Type checks pass (`mypy .`)
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] Branch is up to date with `develop`

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
How has this been tested?

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. **Automated checks** run on all PRs (CI/CD)
2. **Code review** by at least one maintainer
3. **Address feedback** and push updates
4. **Merge** after approval

## 🎯 Areas for Contribution

### High Priority
- [ ] Increase test coverage to 80%+
- [ ] Add integration tests for agent workflows
- [ ] Improve error handling and logging
- [ ] Add backtesting framework
- [ ] Performance optimization

### Medium Priority
- [ ] Add more technical indicators
- [ ] Improve documentation with examples
- [ ] Add API endpoint wrapper
- [ ] Create web UI dashboard

### Good First Issues
Look for issues labeled `good first issue` or `help wanted` in the GitHub issues.

## 📚 Additional Resources

- [Python Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Pytest Documentation](https://docs.pytest.org/)
- [LangChain Documentation](https://python.langchain.com/)

## 💬 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and general discussion
- **Pull Request Comments**: For code-specific questions

## 📜 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

Thank you for contributing! 🎉
