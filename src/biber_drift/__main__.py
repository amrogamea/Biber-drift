"""Allows ``python -m biber_drift ...``."""

import sys

from .cli import main

sys.exit(main())
