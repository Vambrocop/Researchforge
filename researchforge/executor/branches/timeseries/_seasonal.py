"""Seasonal-period detection shared by the timeseries family.

`_periodogram_period` is also imported by branches/forecasting.py and by
tests/test_seasonal_detection.py, so it is re-exported from this package's __init__ to
keep `from ...branches.timeseries import _periodogram_period` working. (By the layering
convention a CROSS-family helper belongs in executor/_helpers/; moving it there changes
forecasting.py's import too, so that is left as a separate, deliberate step.)
"""

from __future__ import annotations

def _periodogram_period(x, n):
    """Dominant seasonal period via the periodogram, or None if no SIGNIFICANT periodicity.
    Linearly detrends first (a trend's low-frequency power otherwise dominates), requires >=3
    cycles (period <= n/3), and applies Fisher's g-test (alpha=0.05) so pure noise/trend -> None."""
    import numpy as np

    x = np.asarray(x, dtype=float)
    idx = np.arange(n)
    c = np.polyfit(idx, x, 1)        # remove linear trend
    x = x - (c[0] * idx + c[1])
    if np.std(x) == 0:
        return None
    power = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n)
    mask = freqs >= 3.0 / n          # candidate seasonal freqs (period <= n/3)
    if not mask.any():
        return None
    pm = power[mask]
    m = len(pm)
    if m < 2 or pm.sum() <= 0:
        return None
    g = float(pm.max() / pm.sum())                    # Fisher's g statistic
    g_crit = 1.0 - (0.05 / m) ** (1.0 / (m - 1))      # alpha=0.05 critical value
    if g <= g_crit:                                   # no significant periodicity
        return None
    freq = freqs[mask][int(np.argmax(pm))]
    if freq <= 0:
        return None
    per = int(round(1.0 / freq))
    return per if 2 <= per <= n // 3 else None
