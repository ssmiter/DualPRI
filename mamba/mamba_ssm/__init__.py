"""Vendored Mamba 2.2.2 kernels used by iSCALE.

Only the custom state space operations are imported by the public iSCALE
entry point. High-level language-model classes remain available from their
individual modules but are not imported eagerly here.
"""

from __future__ import annotations

import sys


__version__ = "2.2.2"

# The upstream sources use absolute ``mamba_ssm`` imports internally. Register
# this vendored package under that name so those imports resolve locally without
# requiring a second installation of mamba-ssm.
sys.modules.setdefault("mamba_ssm", sys.modules[__name__])
