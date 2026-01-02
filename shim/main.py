"""FastAPI entry point for the LMStudio-Ollama shim.

The implementation lives in `main.py`; this file is a thin wrapper that
re-exports the same symbols for backward compatibility.
"""

from main import app, run

__all__ = ["app", "run"]

if __name__ == "__main__":
    run()
