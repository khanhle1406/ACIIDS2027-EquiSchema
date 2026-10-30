"""Paired prediction audit; seeds are sampling units, ordered pairs are not."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def jaccard(left: set[tuple], right: set[tuple]) -> float:
    """Compare deduplicated logical facts, including the empty-context case."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def esd(
    original: NDArray,
    view: NDArray,
    clusters: NDArray | None = None,
    classification: bool = True,
    scale: float = 1.0,
    nboot: int = 2000,
) -> dict[str, float]:
    """Paired seed-cluster bootstrap, conditional on the observed database.

    Multinomial seed weights are shared across views and both estimands.
    Original equal-seed pairs are always excluded, including duplicate bootstrap
    draws of a seed. Entity clusters are resampled with all their rows intact.
    Five fitted seeds give limited training-distribution uncertainty coverage.
    """
    a, b = np.asarray(original, float), np.asarray(view, float)
    if a.ndim != 2 or a.shape != b.shape or min(a.shape) < 1 or a.shape[0] < 2:
        raise ValueError("Need aligned [at least two seeds, nonempty entities] arrays")
    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Predictions must be finite and scale positive")
    if nboot < 2:
        raise ValueError("At least two bootstrap replicates required")
    n, m = a.shape
    if classification:
        a, b = a >= 0.5, b >= 0.5
        same = (a[:, None, :] != a[None, :, :]).astype(float)
        cross = (a[:, None, :] != b[None, :, :]).astype(float)
        paired = np.mean(a != b)
    else:
        same = np.abs(a[:, None, :] - a[None, :, :]) / scale
        cross = np.abs(a[:, None, :] - b[None, :, :]) / scale
        paired = np.mean(np.abs(a - b)) / scale
    mask = ~np.eye(n, dtype=bool)
    delta = cross - same
    labels = np.arange(m) if clusters is None else np.asarray(clusters)
    if labels.shape != (m,):
        raise ValueError("Cluster vector must align with prediction entities")
    _, inverse = np.unique(labels, return_inverse=True)
    k = int(inverse.max()) + 1
    rng = np.random.default_rng(20261028)
    draws = np.empty(nboot)
    for i in range(nboot):
        weights = rng.multinomial(n, np.full(n, 1 / n))
        while np.count_nonzero(weights) < 2:
            weights = rng.multinomial(n, np.full(n, 1 / n))
        pairs = weights[:, None] * weights[None, :] * mask
        entity_weights = rng.multinomial(k, np.full(k, 1 / k))[inverse]
        draws[i] = np.einsum("ij,ije,e->", pairs, delta, entity_weights) / (
            pairs.sum() * entity_weights.sum()
        )
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_seed_ind": float(same[mask].mean()),
        "delta_view_ind": float(cross[mask].mean()),
        "esd_ind": float(delta[mask].mean()),
        "esd_ci_lo": float(lo),
        "esd_ci_hi": float(hi),
        "delta_view_paired": float(paired),
    }
