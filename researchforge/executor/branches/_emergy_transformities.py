"""Curated, SOURCED library of PUBLIC emergy transformities (UEVs) — pure data/util.

EMERGY (Odum) accounting expresses every input flow on a common basis — the
**solar emjoule (sej)** — by multiplying a flow's quantity by its **transformity**
(a.k.a. Unit Emergy Value, UEV): the available solar energy directly + indirectly
required to make one unit of that flow. A transformity in sej/J converts an ENERGY
flow; one in sej/g converts a MASS flow; one in sej/$ converts a MONETARY flow.

This module is PURE DATA + a lookup helper. It defines **no @register handler** and
imports nothing heavy, so it is safe to import anywhere. The branch
``resource_emergy.py`` consumes it; tests pin against the exact values below.

⚠ HONESTY / UNCERTAINTY (read this) ⚠
-------------------------------------
Transformities are PUBLIC scientific *approximations*, not physical constants. They:
  * depend on the **emergy baseline** of the global biosphere (the total annual solar
    emergy driving Earth). Values below are stated on the classic **9.26E24 sej/yr
    (Odum 1996)** baseline. Later revisions (Brown & Ulgiati 2010; Campbell) use
    **12.0E24** or **15.2E24 sej/yr** — rescaling every transformity by the baseline
    ratio (×~1.30 or ×~1.64). NEVER mix values from different baselines in one study.
  * carry LARGE uncertainty (often a factor of 2 or more) and are method- and
    context-dependent (system boundary, allocation, data vintage). The figures here
    are round, ILLUSTRATIVE public values for screening / teaching — verify against a
    primary source (Odum 1996/2000; Brown & Ulgiati; the National Environmental
    Accounting Database, NEAD) for any real study.
  * embed a modelling choice in their R/N/F default category (renewable /
    non-renewable / purchased) — see ``category`` per entry; override via the branch's
    ``config['categories']`` when your system boundary differs.

Sources (general): Odum H.T. (1996) *Environmental Accounting: Emergy and
Environmental Decision Making*, Wiley; Brown M.T. & Ulgiati S. (2004, 2010) emergy
algebra & baseline revisions; Odum, Brown & Brandt-Williams (2000) *Folio #1*;
NEAD (National Environmental Accounting Database, Univ. of Florida).
"""

from __future__ import annotations

# The emergy baseline these library values are stated on (disclose everywhere).
BASELINE = "9.26E24 sej/yr (Odum 1996 geobiosphere baseline)"

DISCLAIMER = (
    "转换率(transformity/UEV)为公开近似值(Odum/Brown/NEAD)，依赖能值基线"
    f"（本库统一采用 {BASELINE}；Brown 2010 修订基线 12.0E24/15.2E24 sej/yr 需按比例换算，"
    "切勿跨基线混用）且不确定性大（常达 2 倍以上）、随系统边界/分配/数据年份而变，"
    "务必按你的研究核实主文献。R/N/F 类别为默认建模选择，可用 config 覆盖。"
)


# --------------------------------------------------------------------------- #
# The library. Each entry:
#   value      : transformity (sej per `unit`), on the BASELINE above
#   unit       : "sej/J" | "sej/g" | "sej/$"  (what one unit of the flow is measured in)
#   category   : default R/N/F  (R=renewable, N=non-renewable, F=purchased/feedback)
#   source     : short citation
#   baseline   : the baseline the value is stated on (== BASELINE here)
#   keywords   : lower-case substrings; a column name CONTAINING any of them matches
#
# R/N/F classification convention (STOP-AND-REPORT — standard Odum/EW convention):
#   R (renewable)     : solar, wind, rain (chemical & geopotential), river/runoff,
#                       water (ambient/freshwater), waves, tide, earth heat
#   N (non-renewable) : topsoil / soil organic matter (eroded faster than formed),
#                       groundwater (mined), fossil fuels, minerals, metals ore
#   F (purchased / feedback / imported, from the economy): electricity, refined fuels,
#                       steel/cement/fertilizer/plastic & manufactured materials,
#                       human labor, services, money
# (fossil fuels & soil are NON-renewable storages → N; electricity & manufactured
#  materials & labor are economic feedbacks bought from outside the system → F. This
#  is the convention in Odum 1996 / Brown & Ulgiati; see also Eurostat/NEAD usage.)
# --------------------------------------------------------------------------- #
TRANSFORMITIES: dict[str, dict] = {
    # ---- Renewable environmental energy inputs (R), energy basis sej/J ----------
    "solar_radiation": {
        "value": 1.0, "unit": "sej/J", "category": "R",
        "source": "Odum 1996 (solar transformity = 1 by definition)",
        "baseline": BASELINE,
        "keywords": ("solar", "sunlight", "insolation", "radiation"),
    },
    "wind": {
        "value": 2.45e3, "unit": "sej/J", "category": "R",
        "source": "Odum 1996; Odum et al. 2000 Folio #1 (wind kinetic energy)",
        "baseline": BASELINE,
        "keywords": ("wind",),
    },
    "rain_chemical": {
        "value": 1.82e4, "unit": "sej/J", "category": "R",
        "source": "Odum 1996; Odum et al. 2000 (rain chemical potential energy)",
        "baseline": BASELINE,
        "keywords": ("rain", "precip", "rainfall"),
    },
    "rain_geopotential": {
        "value": 1.05e4, "unit": "sej/J", "category": "R",
        "source": "Odum 1996 (rain geopotential / runoff potential energy)",
        "baseline": BASELINE,
        "keywords": ("geopotential", "runoff_energy", "elevation"),
    },
    "river_water": {
        "value": 4.85e4, "unit": "sej/J", "category": "R",
        "source": "Odum 1996 (river chemical potential energy)",
        "baseline": BASELINE,
        "keywords": ("river", "stream", "runoff", "surface_water"),
    },
    "water": {
        "value": 4.85e4, "unit": "sej/J", "category": "R",
        "source": "Odum 1996 (freshwater chemical potential energy, ambient)",
        "baseline": BASELINE,
        "keywords": ("water", "irrigation", "freshwater", "rainwater"),
    },
    "waves": {
        "value": 5.1e4, "unit": "sej/J", "category": "R",
        "source": "Odum 1996 (wave energy)",
        "baseline": BASELINE,
        "keywords": ("wave",),
    },
    "tide": {
        "value": 7.4e4, "unit": "sej/J", "category": "R",
        "source": "Odum 1996 (tidal energy)",
        "baseline": BASELINE,
        "keywords": ("tide", "tidal"),
    },
    "earth_heat": {
        "value": 1.2e4, "unit": "sej/J", "category": "R",
        "source": "Odum 1996 (deep heat / geothermal flux)",
        "baseline": BASELINE,
        "keywords": ("geothermal", "earth_heat", "deep_heat"),
    },

    # ---- Non-renewable storages (N) -------------------------------------------
    "topsoil": {
        "value": 7.4e4, "unit": "sej/J", "category": "N",
        "source": "Odum 1996; Brandt-Williams 2002 Folio #4 (topsoil organic matter, energy basis)",
        "baseline": BASELINE,
        "keywords": ("topsoil", "soil_organic", "soil_loss", "soil_erosion", "soil"),
    },
    "groundwater": {
        "value": 2.5e5, "unit": "sej/J", "category": "N",
        "source": "Buenfil 2001; Odum 1996 (mined groundwater, energy basis)",
        "baseline": BASELINE,
        "keywords": ("groundwater", "aquifer", "well_water"),
    },
    "natural_gas": {
        "value": 8.05e4, "unit": "sej/J", "category": "N",
        "source": "Odum 1996; Brown & Ulgiati 2004 (natural gas, energy basis)",
        "baseline": BASELINE,
        "keywords": ("natural_gas", "naturalgas", "gas", "methane", "lng"),
    },
    "oil": {
        "value": 9.06e4, "unit": "sej/J", "category": "N",
        "source": "Odum 1996; Brown & Ulgiati 2004 (crude oil / refined fuel, energy basis)",
        "baseline": BASELINE,
        "keywords": ("oil", "crude", "petroleum", "diesel", "gasoline", "fuel"),
    },
    "coal": {
        "value": 6.69e4, "unit": "sej/J", "category": "N",
        "source": "Odum 1996; Brown & Ulgiati 2004 (coal, energy basis)",
        "baseline": BASELINE,
        "keywords": ("coal", "lignite", "anthracite"),
    },

    # ---- Purchased / feedback inputs from the economy (F) ----------------------
    "electricity": {
        "value": 2.77e5, "unit": "sej/J", "category": "F",
        "source": "Odum 1996; Brown & Ulgiati 2004 (grid electricity, energy basis)",
        "baseline": BASELINE,
        "keywords": ("electric", "electricity", "power", "kwh", "grid"),
    },
    "steel": {
        "value": 4.13e9, "unit": "sej/g", "category": "F",
        "source": "Odum 1996; Brown & Buranakarn 2003 (steel, mass basis)",
        "baseline": BASELINE,
        "keywords": ("steel", "iron"),
    },
    "cement": {
        "value": 3.04e9, "unit": "sej/g", "category": "F",
        "source": "Brown & Buranakarn 2003; Odum 1996 (cement, mass basis)",
        "baseline": BASELINE,
        "keywords": ("cement", "concrete"),
    },
    "nitrogen_fertilizer": {
        "value": 6.38e9, "unit": "sej/g", "category": "F",
        "source": "Odum 1996; Brandt-Williams 2002 Folio #4 (N fertilizer, mass basis)",
        "baseline": BASELINE,
        "keywords": ("fertilizer", "fertiliser", "nitrogen", "urea", "_n_"),
    },
    "phosphorus_fertilizer": {
        "value": 6.88e9, "unit": "sej/g", "category": "F",
        "source": "Odum 1996; Brandt-Williams 2002 Folio #4 (P fertilizer, mass basis)",
        "baseline": BASELINE,
        "keywords": ("phosph", "p2o5"),
    },
    "plastic": {
        "value": 9.0e9, "unit": "sej/g", "category": "F",
        "source": "Brown & Buranakarn 2003 (plastics, mass basis)",
        "baseline": BASELINE,
        "keywords": ("plastic", "polymer", "pvc", "polyeth"),
    },
    "machinery_metal": {
        "value": 6.7e9, "unit": "sej/g", "category": "F",
        "source": "Odum 1996 (machinery / manufactured metal goods, mass basis)",
        "baseline": BASELINE,
        "keywords": ("machinery", "machine", "equipment", "metal_goods"),
    },
    "labor": {
        "value": 7.38e6, "unit": "sej/J", "category": "F",
        "source": "Odum 1996 (human labor, energy basis; varies strongly with society)",
        "baseline": BASELINE,
        "keywords": ("labor", "labour", "work_energy", "human_energy", "workforce"),
    },
    "services": {
        "value": 1.0e12, "unit": "sej/$", "category": "F",
        "source": "Odum 1996 (services via emergy-money ratio; ILLUSTRATIVE, country/year-specific)",
        "baseline": BASELINE,
        "keywords": ("service", "services"),
    },
    "money": {
        "value": 1.0e12, "unit": "sej/$", "category": "F",
        "source": "Odum 1996 (emergy-money ratio EMR; ILLUSTRATIVE, country/year-specific)",
        "baseline": BASELINE,
        "keywords": ("money", "cost", "expenditure", "dollars", "usd", "capital"),
    },
}


def lookup_transformity(name: str):
    """Case-insensitive keyword-substring lookup of a transformity for a column name.

    Returns ``(value, unit, category, meta)`` where ``meta`` is the full library dict
    (including ``source``/``baseline``/``keywords`` and the canonical library key under
    ``meta['key']``), or ``None`` if no library entry matches.

    Matching: the column ``name`` (lower-cased) is scanned for each entry's keyword
    substrings; the FIRST library entry with any keyword contained in the name wins.
    To keep matching deterministic regardless of dict iteration nuances, entries are
    tried in their definition order and, within an entry, the LONGEST keyword that
    matches is what we record. (Specific names like ``natural_gas`` therefore beat the
    broad ``gas`` because ``natural_gas`` is its own earlier entry.)
    """
    if not isinstance(name, str):
        return None
    low = name.strip().lower()
    if not low:
        return None
    for key, meta in TRANSFORMITIES.items():
        for kw in meta["keywords"]:
            if kw in low:
                full = dict(meta)
                full["key"] = key
                return (
                    float(meta["value"]),
                    str(meta["unit"]),
                    str(meta["category"]),
                    full,
                )
    return None
