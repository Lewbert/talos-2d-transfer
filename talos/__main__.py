"""Entry point: python -m talos [--sim]"""

from __future__ import annotations

import argparse
import sys

from talos.app import TALOSApplication
from talos.bootstrap import bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(prog="talos", description="TALOS microscope control")
    parser.add_argument("--sim", action="store_true",
                        help="Run with simulated devices (no hardware access)")
    args = parser.parse_args()

    qapp = bootstrap()
    application = TALOSApplication(qapp, sim=args.sim)
    try:
        return application.run()
    finally:
        application.shutdown()


if __name__ == "__main__":
    sys.exit(main())
