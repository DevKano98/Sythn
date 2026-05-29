# Contributing to SynthID Remover

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

### 1. Clone and setup environment

```bash
git clone <repo-url>
cd synthid-remover-app
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if available
```

### 2. Download test models

```bash
python download_models.py
```

### 3. Run tests

```bash
python -m pytest tests/
# or
python tests/test_basic.py
```

## Project Structure

```
src/
├── models/       # Model loading and inference
├── pipeline/     # Processing stages and orchestration
├── gui/          # User interface components
└── utils/        # Image processing, helpers, validators
```

## How to Contribute

### Reporting Bugs

1. Check if the issue already exists
2. Include:
   - Python version
   - Operating system
   - GPU/CPU mode
   - Error message / stack trace
   - Steps to reproduce

### Suggesting Features

1. Open an issue with the "feature request" label
2. Describe the use case
3. Explain expected behavior

### Code Contributions

#### Adding a Pipeline Stage

1. Create class in `src/pipeline/stages.py`
2. Follow the naming convention: `StageN_DescriptiveName`
3. Implement `process()` method returning a dict
4. Add to `SynthIDPipeline._setup_stages()`
5. Add execution in `SynthIDPipeline.run()`
6. Update progress callback
7. Add tests

Example:
```python
class Stage13_NewFeature:
    def process(self, image: np.ndarray) -> dict:
        # Your processing logic
        return {"result": processed_image}
```

#### Adding a Model

1. Add entry to `config.py` MODELS dict
2. Add loader in `src/models/loader.py`
3. Add inference wrapper in `src/models/inference.py`
4. Update model status in GUI
5. Update documentation

#### GUI Improvements

1. Widgets go in `src/gui/widgets.py`
2. Dialogs go in `src/gui/dialogs.py`
3. Main app logic in `src/gui/app.py`
4. Follow CustomTkinter patterns
5. Ensure keyboard accessibility

### Code Style

- **Python**: PEP 8
- **Type hints**: Use where possible
- **Docstrings**: Google style
- **Imports**: Group by standard lib, third-party, local

### Testing

- Add tests to `tests/`
- Run full test suite before submitting
- Ensure GUI tests work on your platform

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes with clear commits
4. Add/update tests
5. Update documentation
6. Submit PR with description

## Areas for Contribution

### Good First Issues
- [ ] Improve documentation
- [ ] Add more image format support
- [ ] Better error messages
- [ ] UI theme improvements

### Performance
- [ ] Faster preprocessing
- [ ] Model quantization options
- [ ] Memory optimization
- [ ] Batch processing improvements

### Features
- [ ] Video frame processing
- [ ] Plugin system
- [ ] Cloud model loading
- [ ] Mobile/remote UI

## Questions?

Open an issue or discussion. We're happy to help!
