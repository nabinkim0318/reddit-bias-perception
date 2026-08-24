"""Multiple-testing helpers.

Benjamini-Hochberg FDR is the project convention for the family of
emotion-level omnibus tests. BH does not eliminate researcher degrees of
freedom.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

MULTIPLICITY_METHOD = "benjamini_hochberg"


class MultiplicityError(ValueError):
    """Raised when p-values cannot be adjusted."""


def _is_missing_p(value: Any) -> bool:
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    return math.isnan(number)


def benjamini_hochberg(
    p_values: Sequence[Optional[float]],
) -> list[Optional[float]]:
    """Return BH/FDR q-values in the original input order.

    Null/unavailable p-values are preserved as ``None`` and are excluded from
    the correction family size ``m``. Adjusted values are clipped to ``[0, 1]``
    and made monotone non-decreasing in the ranked raw p-values.
    """
    indexed: list[tuple[int, float]] = []
    for i, raw in enumerate(p_values):
        if _is_missing_p(raw):
            continue
        p_value = float(raw)
        if p_value < 0.0 or p_value > 1.0:
            raise MultiplicityError(
                f"p-value at index {i} is outside [0, 1]: {p_value}"
            )
        indexed.append((i, p_value))

    q_values: list[Optional[float]] = [None] * len(p_values)
    m = len(indexed)
    if m == 0:
        return q_values

    indexed.sort(key=lambda item: (item[1], item[0]))
    raw_adjusted = [
        min(1.0, p * m / rank) for rank, (_, p) in enumerate(indexed, start=1)
    ]
    for i in range(m - 2, -1, -1):
        raw_adjusted[i] = min(raw_adjusted[i], raw_adjusted[i + 1])
    for (original_index, _), adjusted in zip(indexed, raw_adjusted):
        q_values[original_index] = float(adjusted)
    return q_values


def family_size(p_values: Sequence[Optional[float]]) -> int:
    return sum(0 if _is_missing_p(p) else 1 for p in p_values)
