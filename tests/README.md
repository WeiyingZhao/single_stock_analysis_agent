# Test Suite

Comprehensive test suite for the Stock Analysis Multi-Agent System.

## Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── unit/                    # Fast, isolated unit tests
│   ├── test_config.py
│   ├── test_models.py
│   └── test_technical_indicators.py
├── integration/             # Integration tests (agents, workflows)
└── fixtures/                # Test data and mock responses
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage
```bash
pytest --cov=. --cov-report=html
```

### Run only unit tests
```bash
pytest tests/unit/ -v
```

### Run only fast tests (exclude slow tests)
```bash
pytest -m "not slow"
```

### Run specific test file
```bash
pytest tests/unit/test_config.py -v
```

### Run specific test
```bash
pytest tests/unit/test_config.py::TestConfig::test_config_validation_with_valid_values -v
```

## Test Markers

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.slow` - Slow tests (can be skipped)
- `@pytest.mark.llm` - Tests that call actual LLM APIs (expensive)

## Writing Tests

### Unit Test Example

```python
import pytest

@pytest.mark.unit
def test_my_function():
    from mymodule import my_function

    result = my_function(input_value)
    assert result == expected_value
```

### Using Fixtures

```python
@pytest.mark.unit
def test_with_mock_data(mock_stock_data):
    # mock_stock_data is provided by conftest.py
    assert len(mock_stock_data) > 0
```

### Testing Exceptions

```python
import pytest
from pydantic import ValidationError

def test_invalid_input():
    from models.event_profile import EventProfile

    with pytest.raises(ValidationError):
        EventProfile(invalid_field="value")
```

## Coverage Goals

- **Overall**: 80%+ coverage
- **Critical paths**: 95%+ coverage
- **Models**: 100% coverage (easy with Pydantic)
- **Utilities**: 90%+ coverage
- **Agents**: 70%+ coverage (harder due to LLM calls)

## Continuous Integration

Tests run automatically on:
- Every push to main/develop
- Every pull request
- Using GitHub Actions (see `.github/workflows/ci.yml`)

## Best Practices

1. **Keep tests fast** - Unit tests should run in milliseconds
2. **Use mocks** - Don't call external APIs in unit tests
3. **One assertion per test** - Tests should be focused
4. **Descriptive names** - Test names should describe what they test
5. **Arrange-Act-Assert** - Clear test structure
6. **Fixtures for reuse** - Share common test data via fixtures

## Mocking External Services

### Mock LLM calls

```python
from unittest.mock import Mock, patch

@patch('langchain_google_genai.ChatGoogleGenerativeAI')
def test_agent_with_mock_llm(mock_llm):
    mock_llm.return_value.invoke.return_value = {
        'messages': [Mock(content="Mocked response")]
    }
    # Your test here
```

### Mock Yahoo Finance

```python
@patch('yfinance.Ticker')
def test_stock_data_fetch(mock_ticker):
    mock_ticker.return_value.history.return_value = mock_dataframe
    # Your test here
```

## Future Test Additions

- [ ] Integration tests for multi-agent workflows
- [ ] Performance benchmarks
- [ ] Load tests for vector database
- [ ] End-to-end tests with real (cached) data
- [ ] Property-based tests with Hypothesis
