"""R-bridge estimator delegators (lavaan / QCA / gstat / spdep / metafor / mgcv /
dbarts / ... via executor.rbridge), split by domain to stay under the 1500-line
module guardrail. Re-exported by run.py — this __init__ is the stable import surface
(``from ..._helpers.r_backends import _xxx`` keeps working)."""

from __future__ import annotations

from .sem import _sem_via_lavaan
from .configurational import _csqca_via_r, _fsqca_via_r, _panel_qca_via_r, _cna_via_r, _qca_necessity_via_r
from .spatial import _kriging_via_r, _spatial_reg_via_r
from .econometrics import _dynamic_gmm_via_r
from .ecology import _diff_abundance_aldex2_via_r
from .survival import _jm_via_r
from .meta import _meta_via_r, _meta_regression_via_r
from .causal import _cic_via_r, _gsynth_via_r
from .mixed_models import _glmm_via_r, _gamm_via_r, _bart_via_r, _gam_via_r
from .efficiency import _sfa_via_r

__all__ = [
    "_bart_via_r",
    "_cic_via_r",
    "_cna_via_r",
    "_csqca_via_r",
    "_diff_abundance_aldex2_via_r",
    "_dynamic_gmm_via_r",
    "_fsqca_via_r",
    "_gam_via_r",
    "_gamm_via_r",
    "_glmm_via_r",
    "_gsynth_via_r",
    "_jm_via_r",
    "_kriging_via_r",
    "_meta_regression_via_r",
    "_meta_via_r",
    "_panel_qca_via_r",
    "_qca_necessity_via_r",
    "_sem_via_lavaan",
    "_sfa_via_r",
    "_spatial_reg_via_r",
]
