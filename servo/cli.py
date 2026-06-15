#!/usr/bin/env python3

import argparse
import logging
import traceback
from . import logger
from . import run


def build_parser():
    """
    CLI parser with global logging options and subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="servo",
        description="Servo command-line tool",
    )

    # -------- Logging options --------
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set global log level",
    )

    parser.add_argument(
        "--log-file",
        default=None,
        help="Override log file location",
    )

    # -------- Subcommands --------
    subparsers = parser.add_subparsers(dest="command", required=True)

    # calib
    calib_p = subparsers.add_parser("calib", help="Run calibration mode")
    calib_p.add_argument("-nc", "--nocam", action="store_true", help="Disable camera")
    calib_p.add_argument("-nv", "--noviewer", action="store_true", help="Disable viewer")

    # run
    run_p = subparsers.add_parser("run", help="Run main execution mode")
    run_p.add_argument("-nc", "--nocam", action="store_true", help="Disable camera")
    run_p.add_argument("-nv", "--noviewer", action="store_true", help="Disable viewer")

    return parser


def main():
    """
    CLI entry point.
    Initializes logging, then dispatches to subcommands.
    """
    parser = build_parser()
    args = parser.parse_args()

    try:
        # ✅ Init logging FIRST (critical)
        logger.init_logging(
            level=args.log_level,
            log_file=args.log_file,
        )

        log = logging.getLogger(__name__)
        log.info(f"CLI args: {args}")

        # ✅ Dispatch
        if args.command == "calib":
            return run.run(mode="calib", nocam=args.nocam, noviewer=args.noviewer)

        elif args.command == "run":
            return run.run(mode="loop", nocam=args.nocam, noviewer=args.noviewer)

    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")

    except Exception:
        # ✅ fallback safe (no recursion, no logger dependency)
        print("Fatal error:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
