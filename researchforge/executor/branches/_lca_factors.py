"""Small CURATED public emission / intensity factor library for footprint_analysis.

This is a PURE DATA + LOOKUP module (no @register handler, no import side effects).
The branch auto-discovery in ``branches/__init__.py`` will import it; that is fine —
importing it only defines constants and functions, it touches no global state and
registers no branch.

HONESTY (read this before trusting any number here)
---------------------------------------------------
This is a *small, curated set of WELL-KNOWN PUBLIC factors* (GWP100 characterisation
factors from the IPCC, published fuel-combustion CO2 factors from EPA/DEFRA, and a
world-average grid carbon intensity from IEA/Ember). These are public scientific
constants, NOT licensed life-cycle-inventory database content (ecoinvent / GaBi /
Sphera etc. require a licence and are NOT reproduced here). Every value is a GENERIC
approximation: real footprints depend on your region, year, technology mix, system
boundary and allocation choices. ALWAYS verify against an authoritative source for
your own scenario and reporting year before using these results for any claim.

WHAT IS COVERED
---------------
* ``carbon`` (the well-supported category):
    - GWP100 characterisation factors (kg CO2e per kg of gas) — IPCC AR6.
    - Fuel combustion CO2 factors (kg CO2 per physical unit) — DEFRA/EPA.
    - Grid electricity carbon intensity (kg CO2 per kWh) — IEA/Ember world average
      plus two ILLUSTRATIVE low-carbon / coal-heavy anchors.
* ``water`` and ``energy``: only a few clearly-ILLUSTRATIVE placeholders are
    included so the multi-impact dimension can be exercised; these are coarse
    teaching values, not authoritative footprints — treat as "needs your own
    factors" for any real study.

Each factor record is a dict:
    {"value": float, "unit": str, "source": str, "year": int|str,
     "keywords": [substrings matched case-insensitively against a column name],
     "note": str (optional caveat)}

Matching rule (see ``lookup_factor``): an activity/column name matches a factor if
ANY of the factor's keywords is a case-insensitive SUBSTRING of the (normalised)
name. Within a category, factors are tried in list order and the FIRST match wins,
so more specific keyword sets are listed before generic ones.
"""

from __future__ import annotations

from typing import Optional

# --------------------------------------------------------------------------- #
# Version / disclaimer (surfaced verbatim in the analysis summary)
# --------------------------------------------------------------------------- #
LIBRARY_VERSION = "rf-lca-factors-0.2 (as of 2026-06)"

DISCLAIMER = (
    "小型公开因子库(碳:IPCC AR6 2021 GWP / DEFRA-DESNZ 2025 燃料 / Ember 2026 电网;"
    "水:WFN Mekonnen-Hoekstra 全球均值),非完整 LCA 数据库(ecoinvent/GaBi 等需授权);"
    "因子为通用近似,且无法运行时联网取最新——库带版本号+年份,请按你的场景/年份核实并定期对照官方源刷新。"
)

# --------------------------------------------------------------------------- #
# carbon — GWP100 characterisation factors (kg CO2e per kg of the gas)
#   IPCC AR6 (2021), GWP100. CH4 fossil ~29.8 / non-fossil ~27.9 — we publish the
#   FOSSIL value (~30) as the default and disclose the choice in the note.
# --------------------------------------------------------------------------- #
_GWP100 = [
    {
        "value": 1.0,
        "unit": "kg CO2e / kg CO2",
        "source": "IPCC AR6 (2021) GWP100",
        "year": 2021,
        "keywords": ["co2", "carbon dioxide", "carbon_dioxide"],
        "note": "参考气体，GWP100=1。",
    },
    {
        "value": 29.8,
        "unit": "kg CO2e / kg CH4",
        "source": "IPCC AR6 (2021) GWP100",
        "year": 2021,
        "keywords": ["ch4", "methane"],
        "note": "采用化石源 CH4 GWP100≈29.8（非化石源≈27.9，本库取化石值并披露）。",
    },
    {
        "value": 273.0,
        "unit": "kg CO2e / kg N2O",
        "source": "IPCC AR6 (2021) GWP100",
        "year": 2021,
        "keywords": ["n2o", "nitrous oxide", "nitrous_oxide"],
        "note": "N2O GWP100≈273。",
    },
    {
        "value": 25200.0,
        "unit": "kg CO2e / kg SF6",
        "source": "IPCC AR6 (2021) GWP100",
        "year": 2021,
        "keywords": ["sf6", "sulfur hexafluoride", "sulphur hexafluoride"],
        "note": "SF6 GWP100 极高(≈25200)——'high' 类强温室气体。",
    },
    {
        "value": 1530.0,
        "unit": "kg CO2e / kg HFC-134a",
        "source": "IPCC AR6 (2021) GWP100",
        "year": 2021,
        "keywords": ["hfc134a", "hfc-134a", "hfc_134a", "hfc134", "r134a"],
        "note": "HFC-134a GWP100≈1530(代表性 HFC，'high' 类)；其它 HFC 差异大，请按具体物种核实。",
    },
]

# --------------------------------------------------------------------------- #
# carbon — fuel combustion CO2 factors (kg CO2 per physical unit)
#   DEFRA/EPA published combustion (scope-1) factors. Units stated explicitly per
#   fuel; mixing units is meaningless, so the unit is part of the disclosure.
# --------------------------------------------------------------------------- #
# DEFRA/DESNZ "UK Government GHG Conversion Factors 2025" (published June 2025) —
# CO2e per physical unit (CO2+CH4+N2O combined at IPCC AR5 GWP100, DEFRA convention).
_DEFRA = "DEFRA/DESNZ UK GHG Conversion Factors 2025 (CO2e, AR5 GWP)"
_FUEL_COMBUSTION = [
    {
        "value": 0.18296,
        "unit": "kg CO2e / kWh natural gas (gross CV)",
        "source": _DEFRA, "year": 2025,
        "keywords": ["natural gas", "natural_gas", "naturalgas", "nat gas", "ng"],
        "note": "天然气 0.18296 kgCO2e/kWh(总热值,DEFRA 2025);若列是体积 m³≈2.0 kgCO2e/m³。",
    },
    {
        "value": 2.57082,
        "unit": "kg CO2e / L diesel",
        "source": _DEFRA, "year": 2025,
        "keywords": ["diesel", "gas oil", "gasoil"],
        "note": "柴油 2.57082 kgCO2e/L(DEFRA 2025,平均生物柴油掺混)。",
    },
    {
        "value": 2.06916,
        "unit": "kg CO2e / L petrol",
        "source": _DEFRA, "year": 2025,
        "keywords": ["gasoline", "petrol"],
        "note": "汽油/petrol 2.06916 kgCO2e/L(DEFRA 2025)。",
    },
    {
        "value": 1.55709,
        "unit": "kg CO2e / L LPG",
        "source": _DEFRA, "year": 2025,
        "keywords": ["lpg", "liquefied petroleum"],
        "note": "LPG≈1.557 kgCO2e/L(DEFRA 2025)。",
    },
    {
        "value": 2.40,
        "unit": "kg CO2 / kg coal",
        "source": "IPCC/DEFRA (coal, varies by rank/CV)", "year": 2025,
        "keywords": ["coal"],
        "note": "煤≈2.4 kg CO2/kg(随煤种/热值差异大,取代表值)。",
    },
]

# --------------------------------------------------------------------------- #
# carbon — grid electricity carbon intensity (kg CO2 per kWh)
#   IEA/Ember world average; two illustrative anchors (low-carbon / coal-heavy).
#   The generic "electricity" keyword maps to the WORLD AVERAGE — disclose that
#   the user's own grid factor should replace it.
# --------------------------------------------------------------------------- #
_GRID = [
    {
        "value": 0.05,
        "unit": "kg CO2 / kWh",
        "source": "IEA/Ember (illustrative low-carbon grid)",
        "year": 2023,
        "keywords": ["electricity_lowcarbon", "low carbon electricity",
                     "low_carbon_electricity", "renewable electricity",
                     "green electricity"],
        "note": "示例：低碳电网≈0.05 kg CO2/kWh（说明用，非你的电网实值）。",
    },
    {
        "value": 0.9,
        "unit": "kg CO2 / kWh",
        "source": "IEA/Ember (illustrative coal-heavy grid)",
        "year": 2023,
        "keywords": ["electricity_coal", "coal electricity", "coal_electricity",
                     "coal grid", "coal_power"],
        "note": "示例：高煤占比电网≈0.9 kg CO2/kWh（说明用，非你的电网实值）。",
    },
    {
        "value": 0.458,
        "unit": "kg CO2 / kWh",
        "source": "Ember Global Electricity Review 2026 (2025 world average)",
        "year": 2025,
        "keywords": ["electricity", "grid", "power", "kwh", "elec"],
        "note": "电网用电≈0.458 kg CO2/kWh（2025 世界平均,Ember GER 2026;2024=0.471）；请用你所在电网/年份的实值替换。",
    },
]

# --------------------------------------------------------------------------- #
# water — ILLUSTRATIVE placeholders only (NOT authoritative footprints)
#   Coarse teaching anchors so the multi-impact path can be exercised. For any
#   real study supply your own water-footprint factors.
# --------------------------------------------------------------------------- #
# Real global-average TOTAL (green+blue+grey) water footprints from the Water
# Footprint Network (Mekonnen & Hoekstra 2011 crops / 2012 farm animals). 1 m³/ton =
# 1 L/kg. These are GLOBAL AVERAGES — local/production-system values vary widely.
_WFN = "Mekonnen & Hoekstra, Water Footprint Network (global average, total green+blue+grey)"
_WATER = [
    {"value": 15400.0, "unit": "L water / kg", "source": _WFN, "year": "2012",
     "keywords": ["beef", "cattle meat"], "note": "牛肉，全球均值≈15400 L/kg。"},
    {"value": 10400.0, "unit": "L water / kg", "source": _WFN, "year": "2012",
     "keywords": ["sheep", "mutton", "lamb"], "note": "羊肉。"},
    {"value": 5990.0, "unit": "L water / kg", "source": _WFN, "year": "2012",
     "keywords": ["pork", "pig"], "note": "猪肉。"},
    {"value": 4330.0, "unit": "L water / kg", "source": _WFN, "year": "2012",
     "keywords": ["chicken", "poultry"], "note": "禽肉。"},
    {"value": 3265.0, "unit": "L water / kg", "source": _WFN, "year": "2012",
     "keywords": ["egg"], "note": "鸡蛋。"},
    {"value": 1020.0, "unit": "L water / kg", "source": _WFN, "year": "2012",
     "keywords": ["milk", "dairy"], "note": "牛奶。"},
    {"value": 18900.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["coffee"], "note": "咖啡(焙炒)。"},
    {"value": 10000.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["cotton"], "note": "棉花。"},
    {"value": 2500.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["rice", "paddy"], "note": "稻米。"},
    {"value": 1830.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["wheat"], "note": "小麦。"},
    {"value": 2145.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["soybean", "soya", "soy"], "note": "大豆。"},
    {"value": 1220.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["maize", "corn"], "note": "玉米。"},
    {"value": 1780.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["sugar"], "note": "蔗糖。"},
    {"value": 287.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["potato"], "note": "马铃薯。"},
    {"value": 214.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["tomato"], "note": "番茄。"},
    {"value": 822.0, "unit": "L water / kg", "source": _WFN, "year": "2011",
     "keywords": ["apple"], "note": "苹果。"},
    {"value": 1.0, "unit": "L water / L withdrawal", "source": "direct withdrawal (1:1)",
     "year": "n/a", "keywords": ["water", "withdrawal", "intake"],
     "note": "直接取水/用水量按 1:1 计入水足迹（蓝水）；产品虚拟水请用上面的 WFN 因子。"},
]

# --------------------------------------------------------------------------- #
# energy — ILLUSTRATIVE placeholders only (NOT authoritative footprints)
# --------------------------------------------------------------------------- #
_ENERGY = [
    {
        "value": 1.0,
        "unit": "kWh primary / kWh delivered (illustrative)",
        "source": "ILLUSTRATIVE placeholder (supply your own energy-intensity factor)",
        "year": "n/a",
        "keywords": ["electricity", "energy", "kwh", "power", "elec"],
        "note": "示意占位：交付电能按 1:1 计一次能源；真实一次能源系数随发电结构变化，请用专用因子。",
    },
]

# Category -> ordered list of factor records (first keyword match wins).
LIBRARY: dict[str, list[dict]] = {
    "carbon": _GWP100 + _FUEL_COMBUSTION + _GRID,
    "water": _WATER,
    "energy": _ENERGY,
}

# Categories backed by sourced public factors (carbon: IPCC/EPA/DEFRA/IEA; water:
# Water Footprint Network global averages) vs illustrative-only (energy).
CARBON_ONLY_AUTHORITATIVE = "carbon"
ILLUSTRATIVE_CATEGORIES = ("energy",)


def _normalise(name: str) -> str:
    """Lowercase + collapse separators so 'Natural Gas', 'natural_gas' match alike."""
    s = str(name).strip().lower()
    # treat -, _, ., and runs of whitespace uniformly so substring matching is robust
    for ch in ("-", "_", "."):
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def _kw_norm(kw: str) -> str:
    s = str(kw).strip().lower()
    for ch in ("-", "_", "."):
        s = s.replace(ch, " ")
    return " ".join(s.split())


def lookup_factor(name: str, category: str = "carbon") -> tuple[Optional[float], Optional[dict]]:
    """Match an activity / column name to a library factor.

    Case-insensitive SUBSTRING match of any of a factor's keywords against the
    normalised name. Within the category, factors are tried in list order and the
    FIRST match wins (specific keyword sets are listed before generic ones).

    Returns ``(value, meta)`` where ``meta`` is a dict with keys
    ``value/unit/source/year/keyword/category/note``, or ``(None, None)`` if no
    factor in the (existing) category matches. Unknown category -> (None, None).
    """
    records = LIBRARY.get(str(category).strip().lower())
    if not records:
        return None, None
    norm = _normalise(name)
    if not norm:
        return None, None
    for rec in records:
        for kw in rec.get("keywords", []):
            if _kw_norm(kw) and _kw_norm(kw) in norm:
                meta = {
                    "value": float(rec["value"]),
                    "unit": rec["unit"],
                    "source": rec["source"],
                    "year": rec["year"],
                    "keyword": kw,
                    "category": str(category).strip().lower(),
                    "note": rec.get("note", ""),
                }
                return float(rec["value"]), meta
    return None, None
