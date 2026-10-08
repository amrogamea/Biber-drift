"""Command-line interface.

    biber-drift run --config configs/ao3.yaml [--stages prepare,features] [--set mmd.sample_size=300]
    biber-drift acquire-ao3 --fandom-tag "The Hobbit - J. R. R. Tolkien" --start-year 2009
    biber-drift acquire-acl --years 2000-2026
"""

from __future__ import annotations

import argparse
import logging
import sys

from . import __version__
from .logging_utils import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    from .acquisition import ao3 as ao3_acquisition
    from .acquisition import s2orc_acl as acl_acquisition

    parser = argparse.ArgumentParser(prog="biber-drift", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run the analysis pipeline")
    run.add_argument("--config", required=True, help="YAML config, e.g. configs/ao3.yaml")
    run.add_argument("--stages", default="prepare,features,analyze,report",
                     help="Comma-separated subset of: prepare, features, analyze, report")
    run.add_argument("--data-path", default=None, help="Override data.path from the config")
    run.add_argument("--output-dir", default=None, help="Override output_dir from the config")
    run.add_argument("--name", default=None, help="Override the run name (output sub-folder)")
    run.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                     help="Override any config value, e.g. --set mmd.sample_size=300 (repeatable)")

    ao3 = commands.add_parser("acquire-ao3", help="Scrape AO3 chapters for one fandom tag")
    ao3_acquisition.add_arguments(ao3)
    ao3.set_defaults(handler=ao3_acquisition.run)

    acl = commands.add_parser("acquire-acl", help="Extract ACL papers from S2ORC")
    acl_acquisition.add_arguments(acl)
    acl.set_defaults(handler=acl_acquisition.run)
    return parser


def _run_pipeline(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")  # figures are written to disk; no display needed

    from .config import load_config
    from .pipeline import Pipeline

    overrides = list(args.overrides)
    if args.data_path:
        overrides.append(f"data.path={args.data_path}")
    if args.output_dir:
        overrides.append(f"output_dir={args.output_dir}")
    if args.name:
        overrides.append(f"name={args.name}")
    config = load_config(args.config, overrides)
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    Pipeline(config).run(stages)


def main(argv: list[str] | None = None) -> int:
    from .config import ConfigError
    from .corpus import CorpusSchemaError

    args = _build_parser().parse_args(argv)
    configure_logging(args.log_level)
    try:
        if args.command == "run":
            _run_pipeline(args)
        else:
            args.handler(args)
    except (FileNotFoundError, ConfigError, CorpusSchemaError) as exc:
        logging.getLogger("biber_drift").error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
