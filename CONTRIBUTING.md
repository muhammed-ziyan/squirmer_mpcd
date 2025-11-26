## Contributing to `squirmer_mpcd`

Thank you for your interest in contributing! This project implements a high-performance MPCD/SRD squirmer simulation in Python/Numba. Contributions that improve correctness, performance, documentation, and usability are all welcome.

### Ways to Contribute

- **Bug reports**: File an issue with:
  - A clear description of the problem
  - Steps to reproduce (including command-line arguments)
  - Expected vs. actual behavior
  - Your environment (OS, Python version, NumPy/Numba versions)
- **Feature requests**: Open an issue describing:
  - The scientific or practical motivation
  - A rough idea of the API / configuration needed
- **Pull requests**: Improvements to code, tests, docs, or examples.

### Development Setup

1. Fork the repository on GitHub.
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/squirmer_mpcd.git
   cd squirmer_mpcd
   ```
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -U pytest
   ```

### Coding Guidelines

- Follow **PEP 8** style where practical.
- Prefer clear, explicit code over clever one-liners.
- Keep performance-critical sections (Numba-jitted functions) minimal and well-documented.
- Document key functions and classes with concise docstrings.

### Testing

- Add or update tests in the `tests/` directory when you change behavior.
- Run the test suite before opening a PR:
  ```bash
  python -m pytest tests/
  ```

### Submitting a Pull Request

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes, including tests and documentation where appropriate.
3. Run tests and ensure they pass.
4. Push your branch and open a Pull Request against the main repository.
5. In the PR description, include:
   - What the change does
   - Why it’s useful
   - Any breaking changes or caveats

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT).


