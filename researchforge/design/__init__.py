"""DoE advisory — the "help me design the experiment" mode: question → design.

Unlike the rest of the engine (data → analysis), this generates a randomized
experimental layout *before* any data exists, given the factors/levels/constraints.
"""

from researchforge.design.layout import generate_design, recommend_design

__all__ = ["generate_design", "recommend_design"]
