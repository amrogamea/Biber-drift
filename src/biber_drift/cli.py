"""Command-line interface.

    biber-drift run --config configs/ao3.yaml [--stages prepare,features] [--set mmd.sample_size=300]
    biber-drift acquire-ao3 --fandom-tag "The Hobbit - J. R. R. Tolkien" --start-year 2009
    biber-drift acquire-acl --years 2000-2026
    biber-drift paraphrase sample|download|generate|check|tag|analyze [--model qwen3-8b] [--stage pilot]
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

    paraphrase = commands.add_parser("paraphrase", help="Paraphrase test with open LLMs")
    paraphrase.add_argument("step", choices=["sample", "download", "generate", "check", "tag", "analyze"],
                            help="sample: draw the texts (laptop); download: model weights (HPC login "
                                 "node); generate: paraphrase with one model (HPC GPU job); check: flag "
                                 "refusals, openers, loops and lengths in a generation run; tag: clean, "
                                 "filter and Biber-tag its outputs and the originals (GPU job); analyze: "
                                 "how much each model changes the texts, by year (laptop)")
    paraphrase.add_argument("--config", default="configs/paraphrase.yaml")
    paraphrase.add_argument("--model", default=None,
                            help="generate: a model key or its index in the config (the Slurm array "
                                 "index); download, check, tag, analyze: comma-separated keys (default: every "
                                 "model; tag also takes 'original' and does it first)")
    paraphrase.add_argument("--stage", default="main", choices=["pilot", "main"],
                            help="generate, check: the pilot subset or the full sample")
    paraphrase.add_argument("--name", default=None,
                            help="generate, check, tag, analyze: folder under generations/ (default: the stage)")
    paraphrase.add_argument("--limit", type=int, default=None,
                            help="generate, check, tag: only N texts per corpus, spread over its groups and years")
    paraphrase.add_argument("--tensor-parallel", type=int, default=None, metavar="N",
                            help="generate: split the model over N GPUs, overriding its tensor_parallel_size "
                                 "(e.g. 4 on 10 GB cards; the outputs' settings check ignores it)")
    paraphrase.add_argument("--show", type=int, default=0, metavar="N",
                            help="check: also print N seeded examples per model to the terminal "
                                 "(texts are never written to a file)")
    paraphrase.add_argument("--flag", default=None,
                            help="check --show: only outputs with this flag, e.g. refusal or opener "
                                 "('any' = any flag)")
    paraphrase.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                            help="Override any value of the paraphrase config (repeatable)")
    paraphrase.set_defaults(handler=_run_paraphrase)
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


def _run_paraphrase(args: argparse.Namespace) -> None:
    from .config import ConfigError
    from .paraphrase import (
        ParaphraseChecker, ParaphraseGenerator, ParaphraseTagger, download_models, draw_sample,
        load_paraphrase_config,
    )

    config = load_paraphrase_config(args.config, args.overrides)
    if args.step == "sample":
        draw_sample(config)
    elif args.step == "download":
        failed = download_models(config, args.model.split(",") if args.model else None)
        if failed:
            raise FileNotFoundError(f"Could not download {failed}; see the errors above")
    elif args.step == "check":
        checker = ParaphraseChecker(config, args.stage, args.name,
                                    args.model.split(",") if args.model else None, args.limit)
        checker.run()
        if args.show:
            checker.show(args.show, args.flag)
    elif args.step == "tag":
        ParaphraseTagger(config, args.stage, args.name, args.model.split(",") if args.model else None,
                         args.limit).run()
    elif args.step == "analyze":
        from .paraphrase.analysis import ParaphraseAnalysis    # statsmodels: not in the HPC's vLLM env

        ParaphraseAnalysis(config, args.stage, args.name, args.model.split(",") if args.model else None).run()
    else:
        if args.model is None:
            raise ConfigError("paraphrase generate needs --model (a key or an index in the config)")
        ParaphraseGenerator(config, args.model, args.stage, args.name,
                            args.tensor_parallel).run(limit=args.limit)


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
