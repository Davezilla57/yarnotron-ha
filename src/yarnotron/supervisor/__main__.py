"""python -m yarnotron.supervisor --config supervisor.toml"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import ConfigError, load
from .core import Supervisor
from .server import make_server

log = logging.getLogger("yarnotron.supervisor")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yarnotron-supervisor")
    parser.add_argument("--config", default="~/.yarnotron-supervisor/supervisor.toml")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load(args.config)
    except ConfigError as exc:
        print(f"Config problem: {exc}", file=sys.stderr)
        return 2
    if not config.token:
        log.warning("No token set: anyone on your network can start and stop these services.")
    server = make_server(Supervisor(config))
    log.info("Supervising %d services on %s:%d", len(config.services), config.host, config.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
