"""Entry point for running the shim as a module."""

from main import app, run

__all__ = ["app", "run"]

if __name__ == "__main__":
    run()
