"""CLI entrypoint for running the LM Studio shim."""

import argparse
import uvicorn

import main as app_main
import state
from logging_config import setup_logging


def main() -> None:
    """Run the LM Studio shim server."""
    parser = argparse.ArgumentParser(description="Run the LM Studio shim.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        state.settings.debug = True
        setup_logging(True)

    use_uvloop = app_main.install_uvloop()
    uvicorn.run(
        app_main.app,
        host=args.host,
        port=args.port,
        loop="uvloop" if use_uvloop else "asyncio",
    )


if __name__ == "__main__":
    main()
