"""Package entrypoint.

Allows running:
  python -m dataton_tri_losya_49 infer ...
  python -m dataton_tri_losya_49 experiment ...
"""

from __future__ import annotations

import argparse

from dataton_tri_losya_49.cli.experiment import main as experiment_main
from dataton_tri_losya_49.cli.infer import main as infer_main

_COMMANDS = {
    "infer": infer_main,
    "experiment": experiment_main,
}


def main(argv: list[str] | None = None) -> int:
    """
    Package CLI entrypoint.

    Implements python -m dataton_tri_losya_49 <subcommand> ...
    and dispatches to the actual subcommand handlers.

    Args:
        argv: Optional argv override (useful for tests). If None, argparse uses
            sys.argv.

    Returns:
        Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(prog="dataton_tri_losya_49")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("infer", help="Run inference: CSV -> submission.csv")
    sub.add_parser("experiment", help="Run dev experiment: config -> artifacts")

    ns, rest = parser.parse_known_args(argv)
    return _COMMANDS[ns.cmd](rest)


if __name__ == "__main__":
    raise SystemExit(main())
