# jaxtransform3d

This library is still experimental and a lot of things are subject to change,
e.g., the name. The core idea is to create a version of
[pytransform3d](https://github.com/dfki-ric/pytransform3d)
that is JIT-compiled, executable on GPU, differentiable, and inherently
vectorized.

## Commands

Installation:

```bash
pip install -e .
```

Code formatting:

```bash
black .
```

Linting:

```bash
ruff check
```

Testing:

```bash
pytest
```

## License

The library is released under BSD 3-clause license.
