"""CLI entrypoint for running the LM Studio shim."""

import argparse
import os

import uvicorn

import main as app_main


def main() -> None:
    """Run the LM Studio shim server."""
    parser = argparse.ArgumentParser(description="Run the LM Studio shim.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        os.environ["SHIM_DEBUG"] = "1"

    uvicorn.run(app_main.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
