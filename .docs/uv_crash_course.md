# UV Crash Course

`uv` is an extremely fast Python package installer and resolver, written in Rust, designed as a drop-in replacement for `pip` and `pip-tools`.

## Basic Commands
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```


### Creating a Virtual Environment

```bash
# Create a new virtual environment named .venv
uv venv
```

### Activating the Environment

*   **bash/zsh:** `source .venv/bin/activate`
*   **fish:** `source .venv/bin/activate.fish`
*   **powershell:** `.venv\Scripts\Activate.ps1`

### Installing Packages

```bash
# Install a package (e.g., requests)
uv add requests

# uv seamlessly works with pip
# Install multiple packages
uv pip install requests flask

# Install packages from a requirements file
uv pip install -r requirements.txt

# Install a specific version
uv pip install requests==2.28.1

# Install for development (from pyproject.toml)
uv pip install -e '.[dev]'
```

### Uninstalling Packages

```bash
# Uninstall a package
uv pip uninstall requests
```

### Listing Installed Packages

```bash
# List installed packages
uv pip list
```

### Freezing Dependencies

```bash
# Output installed packages to requirements.txt
uv pip freeze > requirements.txt
```

### Syncing Dependencies

If you have a `requirements.txt` or `pyproject.toml`, you can sync your environment to match it exactly:

```bash
# Just works
uv sync

# Sync based on requirements.txt
uv pip sync requirements.txt

# Sync based on pyproject.toml (including optional dependencies)
uv pip sync pyproject.toml --all-extras
```

## Why use `uv`?

*   **Speed:** It's significantly faster than `pip` due to its Rust implementation and advanced caching/parallelization.
*   **Unified Tool:** Combines functionalities of `pip`, `pip-tools`, `virtualenv`, etc., into one tool.
*   **Resolution:** Has a fast, modern dependency resolver.

For more details, check the [official `uv` documentation](https://github.com/astral-sh/uv). 