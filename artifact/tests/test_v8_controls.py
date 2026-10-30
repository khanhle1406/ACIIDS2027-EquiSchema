"""Substantive regression tests for v8 controlled study implementations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Enforce thread safety
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import lightgbm as _lgb  # noqa: F401
import torch

torch.set_num_threads(1)

import numpy as np
import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = TESTS_DIR.parents[0] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_physical_sampler_study as base
import run_v8_controls as v8
from audit_v8 import esd, jaccard


def test_train_only_normalizer_isolated() -> None:
    """Verify that train normalizer parameters depend strictly on the training partition."""
    n_entities = 100
    rng = np.random.default_rng(20261028)
    raw = rng.uniform(1.0, 100.0, size=(n_entities, 7))

    tr = np.arange(0, 60)
    te = np.arange(60, 100)

    mu1, sd1 = v8.fit_train_normalizer(raw, tr)

    # Mutate test slice radically
    raw_corrupted = raw.copy()
    raw_corrupted[te] = raw_corrupted[te] * 1000.0 + 50000.0

    mu2, sd2 = v8.fit_train_normalizer(raw_corrupted, tr)

    np.testing.assert_allclose(mu1, mu2)
    np.testing.assert_allclose(sd1, sd2)


def test_frozen_normalizer_applied_to_role_swap() -> None:
    """Verify that frozen training parameters are applied to transformed view without refitting."""
    raw_orig = np.array(
        [
            [0.1, 10.0, 2.0, 5.0, 1.0, 0.8, 0.2],
            [0.2, 20.0, 4.0, 8.0, 2.0, 1.2, 0.3],
            [0.3, 30.0, 6.0, 12.0, 3.0, 1.5, 0.4],
        ],
        dtype=float,
    )
    tr = np.array([0, 1])
    mu, sd = v8.fit_train_normalizer(raw_orig, tr)

    # Simulated role swap
    raw_swap = raw_orig.copy()
    raw_swap[:, [1, 2]] = raw_swap[:, [2, 1]]
    raw_swap[:, [3, 4]] = raw_swap[:, [4, 3]]
    raw_swap[:, [5, 6]] = raw_swap[:, [6, 5]]

    normed_swap = v8.apply_normalizer(raw_swap, mu, sd)

    # Column 0 (customer_x) must be untouched
    np.testing.assert_allclose(normed_swap[:, 0], raw_orig[:, 0])

    # Standardized features must use mu and sd from original training split
    expected_col1 = (np.log1p(raw_swap[:, 1]) - mu[0]) / sd[0]
    np.testing.assert_allclose(normed_swap[:, 1], expected_col1, rtol=1e-5)


def test_raw_transaction_role_swap_reaggregation() -> None:
    """Verify that transaction role swap reaggregates incident buyer/seller channels cleanly."""
    cfg = base.DomainCfg(
        "MiniDomain",
        n_customers=20,
        n_merchants=5,
        n_transactions=100,
        heavy_tail=False,
    )
    cust, merchants, tx, _ = base.generate(cfg, seed=123)
    buy, sell = base.incident_lists(tx, len(cust))

    raw_orig, _, _ = v8.sample_view_with_facts(
        "ORIG", 2, cust, merchants, tx, buy, sell
    )

    tx_swapped = v8.swap_transaction_roles(tx)
    assert (tx_swapped["buyer_id"].to_numpy() == tx["seller_id"].to_numpy()).all()
    assert (tx_swapped["seller_id"].to_numpy() == tx["buyer_id"].to_numpy()).all()

    buy_swapped, sell_swapped = base.incident_lists(tx_swapped, len(cust))
    raw_swapped, _, _ = v8.sample_view_with_facts(
        "ORIG", 2, cust, merchants, tx_swapped, buy_swapped, sell_swapped
    )

    # Customer base feature identical
    np.testing.assert_allclose(raw_orig[:, 0], raw_swapped[:, 0])
    # Buyer amount and seller amount swapped
    np.testing.assert_allclose(raw_orig[:, 1], raw_swapped[:, 2])
    np.testing.assert_allclose(raw_orig[:, 2], raw_swapped[:, 1])
    # Buyer count and seller count swapped
    np.testing.assert_allclose(raw_orig[:, 3], raw_swapped[:, 4])
    np.testing.assert_allclose(raw_orig[:, 4], raw_swapped[:, 3])
    # Buyer risk and seller risk swapped
    np.testing.assert_allclose(raw_orig[:, 5], raw_swapped[:, 6])
    np.testing.assert_allclose(raw_orig[:, 6], raw_swapped[:, 5])


def test_oracle_contrast_uses_population_normalizer() -> None:
    """Verify that oracle score uses generator full-population normalizer matching latent signal."""
    cfg = base.DomainCfg(
        "MiniOracleDomain",
        n_customers=50,
        n_merchants=5,
        n_transactions=200,
        heavy_tail=False,
        signal_scale=1.0,
    )
    cust, merchants, tx, labels = base.generate(cfg, seed=456)
    buy, sell = base.incident_lists(tx, len(cust))
    raw_orig, _, _ = v8.sample_view_with_facts(
        "ORIG", 2, cust, merchants, tx, buy, sell
    )

    pop_mu, pop_sd = v8.fit_population_normalizer(raw_orig)
    oracle_scores = v8.compute_oracle_score(raw_orig, pop_mu, pop_sd, cfg.signal_scale)

    # Must match generator labels.oracle_score to floating point tolerance
    np.testing.assert_allclose(oracle_scores, labels.oracle_score.to_numpy(), atol=1e-6)

    # Training scaler would produce a different result
    tr = np.arange(0, 30)
    mu_tr, sd_tr = v8.fit_train_normalizer(raw_orig, tr)
    training_scaled_score = v8.compute_oracle_score(
        raw_orig, mu_tr, sd_tr, cfg.signal_scale
    )
    assert not np.allclose(oracle_scores, training_scaled_score)


def test_serialization_loaders_exercise_role_keyed_vs_anonymous() -> None:
    """Verify that role-keyed loader canonicalizes by role tag while anonymous loader swaps."""
    # Features: [cx, amt_b, amt_s, cnt_b, cnt_s, risk_b, risk_s]
    x_feat = np.array(
        [
            [1.0, 10.0, 20.0, 1.0, 2.0, 0.1, 0.2],
            [2.0, 30.0, 40.0, 3.0, 4.0, 0.3, 0.4],
        ],
        dtype=np.float32,
    )

    payload = v8.serialize_channels_permuted(x_feat)
    # Physical slots are ordered [seller, buyer]
    assert payload.slots[0][0] == "seller"
    assert payload.slots[1][0] == "buyer"

    loaded_role = v8.role_keyed_loader(payload)
    loaded_anon = v8.anonymous_slot_loader(payload)

    # Role-keyed loader restores exact canonical layout
    np.testing.assert_allclose(loaded_role, x_feat)

    # Anonymous loader presents swapped channels
    expected_swapped = np.array(
        [
            [1.0, 20.0, 10.0, 2.0, 1.0, 0.2, 0.1],
            [2.0, 40.0, 30.0, 4.0, 3.0, 0.4, 0.3],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(loaded_anon, expected_swapped)


def test_logical_fact_extraction_and_jaccard_divergence() -> None:
    """Verify that logical fact Jaccard is distinct from single-view full-context fraction."""
    # Customer 0 has 2 buyer transactions, 0 seller transactions
    ids_b = [101, 102]
    ids_s: list[int] = []
    mer_ids = np.zeros(200, dtype=int)
    mer_ids[101] = 3
    mer_ids[102] = 4

    facts_orig_h1 = v8.extract_logical_facts("ORIG", 1, 0, ids_b, ids_s, mer_ids)
    facts_r1_h1 = v8.extract_logical_facts("R1", 1, 0, ids_b, ids_s, mer_ids)

    # In ORIG_H1: struct (<=1) and amount (<=1) are visible
    # In R1_H1: only struct (<=1) is visible (amount is dist 2)
    assert ("tx_struct", "101", "buyer") in facts_orig_h1
    assert ("tx_amount", "101") in facts_orig_h1
    assert ("tx_struct", "101", "buyer") in facts_r1_h1
    assert ("tx_amount", "101") not in facts_r1_h1

    sim = jaccard(facts_orig_h1, facts_r1_h1)
    # Jaccard is strictly less than 1.0 and different from context fractions
    assert 0.0 < sim < 1.0


def test_esd_bootstrap_integration() -> None:
    """Verify integration of audit_v8.esd with paired bootstrap."""
    rng = np.random.default_rng(20261028)
    orig_preds = rng.uniform(0.1, 0.9, size=(5, 50))
    # Add view perturbation
    view_preds = np.clip(orig_preds + rng.normal(0.0, 0.15, size=(5, 50)), 0.0, 1.0)

    res = esd(orig_preds, view_preds, classification=True, nboot=50)
    for field in [
        "delta_seed_ind",
        "delta_view_ind",
        "esd_ind",
        "esd_ci_lo",
        "esd_ci_hi",
        "delta_view_paired",
    ]:
        assert field in res
        assert np.isfinite(res[field])
    assert res["esd_ci_lo"] <= res["esd_ci_hi"]


def test_collapsed_control_raw_and_normalized_invariance() -> None:
    """Verify that collapsed representations are bit-identical and yield exact zero swap response."""
    cfg = base.DomainCfg(
        "MiniCollapsedDomain",
        n_customers=30,
        n_merchants=5,
        n_transactions=120,
        heavy_tail=False,
    )
    cust, merchants, tx, labels = base.generate(cfg, seed=789)
    buy, sell = base.incident_lists(tx, len(cust))
    raw_orig, _, _ = v8.sample_view_with_facts(
        "ORIG", 2, cust, merchants, tx, buy, sell
    )

    tx_swap = v8.swap_transaction_roles(tx)
    buy_swap, sell_swap = base.incident_lists(tx_swap, len(cust))
    raw_swap, _, _ = v8.sample_view_with_facts(
        "ORIG", 2, cust, merchants, tx_swap, buy_swap, sell_swap
    )

    # 1. Raw collapsed arrays must be bit-identical
    raw_coll = v8.build_raw_collapsed(raw_orig)
    raw_swap_coll = v8.build_raw_collapsed(raw_swap)
    np.testing.assert_array_equal(raw_coll, raw_swap_coll)

    # 2. Normalized collapsed arrays must be identical
    tr = np.arange(0, 20)
    te = np.arange(20, 30)
    mu_coll, sd_coll = v8.fit_collapsed_normalizer(raw_coll, tr)
    norm_coll = v8.apply_collapsed_normalizer(raw_coll, mu_coll, sd_coll)
    norm_swap_coll = v8.apply_collapsed_normalizer(raw_swap_coll, mu_coll, sd_coll)
    np.testing.assert_allclose(norm_coll, norm_swap_coll)

    # 3. Symmetric MPNN representations must be identical
    mpnn_coll = v8.build_symmetric_mpnn_channels(norm_coll)
    mpnn_swap_coll = v8.build_symmetric_mpnn_channels(norm_swap_coll)
    np.testing.assert_allclose(mpnn_coll, mpnn_swap_coll)

    # 4. GBDT fit on collapsed produces exact zero swap response
    y = labels.target.to_numpy()
    m_coll, p0 = base.train_gbdt(norm_coll, y, tr, te, seed=11)
    ps = m_coll.predict_proba(norm_swap_coll[te])[:, 1]
    np.testing.assert_allclose(p0, ps, atol=1e-7)


def test_raw_prediction_csv_roundtrip_metadata() -> None:
    """Verify that key parsing preserves full view names (e.g. ORIG_H1) and seeds (e.g. 11)."""
    domain = "Synth-A-uniform"
    test_views = [
        "ORIG_H1",
        "ORIG_H2",
        "R1_H1",
        "R1_H2",
        "R2_H1",
        "R2_H2",
        "R3_H1",
        "R3_H2",
        "R3_H3",
    ]
    test_seeds = [11, 23, 37, 51, 71]

    raw_dict: dict[str, np.ndarray] = {
        f"test_indices_{domain}": np.array([10, 20]),
        f"targets_{domain}": np.array([0, 1]),
    }
    for view in test_views:
        for seed in test_seeds:
            raw_dict[f"pred_{domain}_LightGBM_{view}_s{seed}"] = np.array([0.2, 0.8])
            raw_dict[f"pred_{domain}_TypedMPNN_{view}_s{seed}"] = np.array([0.3, 0.7])

    tabular_predictions: list[dict[str, object]] = []
    prefix = f"pred_{domain}_"
    te = raw_dict[f"test_indices_{domain}"]
    targets = raw_dict[f"targets_{domain}"]

    for key, arr in raw_dict.items():
        if key.startswith(prefix):
            remainder = key[len(prefix) :]
            mdl, rest = remainder.split("_", 1)
            view, seed_str = rest.rsplit("_s", 1)
            s0 = int(seed_str)
            for ent_id, tgt, p in zip(te, targets, arr):
                tabular_predictions.append(
                    {
                        "domain": domain,
                        "model": mdl,
                        "view": view,
                        "seed": s0,
                        "entity_id": int(ent_id),
                        "target": int(tgt),
                        "pred": float(p),
                    }
                )

    df = pd.DataFrame(tabular_predictions)
    assert set(df["view"].unique()) == set(test_views)
    assert set(df["seed"].unique()) == set(test_seeds)
    assert set(df["model"].unique()) == {"LightGBM", "TypedMPNN"}


def test_mini_domain_study_execution(tmp_path: Path) -> None:
    """Run an end-to-end fast mini domain study to ensure all outputs format correctly."""
    cfg = base.DomainCfg(
        "Synth-Mini",
        n_customers=40,
        n_merchants=6,
        n_transactions=150,
        heavy_tail=False,
    )
    perf_df, drift_df, role_df, cov_df, cert, raw_dict, scaler_dict = (
        v8.run_domain_study(cfg, data_seed=999, seeds=(11, 23))
    )

    assert not perf_df.empty
    assert not drift_df.empty
    assert not role_df.empty
    assert not cov_df.empty
    assert "r1_roundtrip_equal" in cert
    assert f"test_indices_{cfg.name}" in raw_dict
    assert "train_mu" in scaler_dict
    assert "collapsed_mu" in scaler_dict

    # Check drift DataFrame has both jaccard and context fractions
    assert "jaccard" in drift_df.columns
    assert "full_context_fraction_ref" in drift_df.columns
    assert "full_context_fraction_view" in drift_df.columns
    assert "esd_ci_lo" in drift_df.columns

    # Check role DataFrame has all three encodings and non-null values
    encodings = set(role_df["encoding"].unique())
    assert {"role-keyed", "anonymous-slot", "collapsed"}.issubset(encodings)
    role_keyed_disagree = role_df[role_df["encoding"] == "role-keyed"][
        "serialization_disagreement"
    ].mean()
    anon_disagree = role_df[role_df["encoding"] == "anonymous-slot"][
        "serialization_disagreement"
    ].mean()
    collapsed_resp = role_df[role_df["encoding"] == "collapsed"]["swap_response"].mean()
    collapsed_disagree = role_df[role_df["encoding"] == "collapsed"][
        "serialization_disagreement"
    ].mean()

    assert role_keyed_disagree == 0.0
    assert anon_disagree > 0.0
    assert collapsed_resp == 0.0
    assert collapsed_disagree == 0.0


def test_r2_preserves_an_unused_dimension_member() -> None:
    """An unreferenced merchant must survive denormalization plus its inverse."""
    import pandas as pd

    merchants = pd.DataFrame({"merchant_id": [0, 1], "merchant_risk": [0.2, 0.8]})
    transactions = pd.DataFrame(
        {
            "transaction_id": [0],
            "buyer_id": [0],
            "seller_id": [1],
            "merchant_id": [0],
            "amount": [3.0],
            "timestamp": pd.to_datetime(["2023-01-01"]),
        }
    )
    from run_v8_controls import base

    report = base.certify(merchants, transactions)
    assert report["r2_unused_dimension_rows"] == 1
    assert report["orig_merchants_sha256"] == report["r2_merchants_sha256"]
