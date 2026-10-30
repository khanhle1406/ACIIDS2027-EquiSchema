"""Regenerate manuscript tables and calibration plot from verified v8 CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("Synth-A-uniform", "Synth-B-heavy-tail")
MODELS = ("LightGBM", "TypedMPNN")


def main() -> None:
    source = ROOT / "results/v8/controlled"
    target = ROOT / "paper/generated"
    target.mkdir(exist_ok=True)
    perf = pd.read_csv(source / "v8_performance_summary.csv")
    drift = pd.read_csv(source / "v8_drift_summary.csv")
    role = pd.read_csv(source / "v8_role_summary.csv")
    coverage = pd.read_csv(source / "v8_coverage_summary.csv")
    rows = []
    states = [
        (r"$S_{\rm low}$", r"R1$_1$, R3$_1$", ".33", "R1_H1"),
        (r"$S_{\rm mid}$", r"$O_1$, R3$_2$", ".67", "ORIG_H1"),
        (r"$S_{\rm full}$", r"$O_2$, R1$_2$, R2$_1$/R2$_2$, R3$_3$", "1.00", "ORIG_H2"),
    ]
    for label, aliases, fraction, view in states:
        cells = []
        for domain in DOMAINS:
            for model in MODELS:
                record = perf[
                    (perf.domain == domain)
                    & (perf.model == model)
                    & (perf.view == view)
                ].iloc[0]
                cells.append(f"{record.auc_mean:.3f}$\\pm${record.auc_std:.3f}")
        rows.append(" & ".join([label, aliases, fraction, *cells]) + r"\\")
    (target / "controlled_performance.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for model in MODELS:
        for i, domain in enumerate(DOMAINS):
            for j, (transition, display) in enumerate(
                [
                    ("mid-to-low (R1)", r"mid$\to$low (R1)"),
                    ("mid-to-full (R2)", r"mid$\to$full (R2)"),
                    ("full-to-mid (R3)", r"full$\to$mid (R3)"),
                ]
            ):
                record = drift[
                    (drift.domain == domain)
                    & (drift.model == model)
                    & (drift.transition == transition)
                ].iloc[0]
                first = r"\multirow{6}{*}{" + model + "}" if i == 0 and j == 0 else ""
                rows.append(
                    f"{first} & {'A' if i == 0 else 'B'} & {display} & {100 * record.delta_seed_ind:.2f} & {100 * record.delta_view_ind:.2f} & {100 * record.esd_ind:.2f} [{100 * record.esd_ci_lo:.2f},{100 * record.esd_ci_hi:.2f}]"
                    + r"\\"
                )
        if model == MODELS[0]:
            rows.append(r"\midrule")
    (target / "controlled_drift.tex").write_text("\n".join(rows) + "\n")
    rows = []
    for model in MODELS:
        for encoding in ("role-keyed", "anonymous-slot", "collapsed"):
            records = [
                role[
                    (role.domain == domain)
                    & (role.model == model)
                    & (role.encoding == encoding)
                ].iloc[0]
                for domain in DOMAINS
            ]
            values = [100 * r.swap_concordance for r in records] + [
                100 * r.serialization_disagreement for r in records
            ]
            rows.append(
                " & ".join([model, encoding, *(f"{v:.2f}" for v in values)]) + r"\\"
            )
    (target / "controlled_role.tex").write_text("\n".join(rows) + "\n")
    real = ROOT / "results/v8/relbench"
    actual = (
        pd.read_csv(real / "evidence/performance.csv")
        .groupby("view")
        .nmae.agg(["mean", "std"])
    )
    verified = json.loads((real / "analysis_verified.json").read_text())
    measure = json.loads((real / "evidence/summary.json").read_text())
    median = json.loads((real / "median_reference.json").read_text())
    real_rows = []
    for view, label in (("original", "Original"), ("R1", "R1")):
        row = actual.loc[view]
        real_rows.append(
            f"{label} GraphSAGE NMAE, mean $\\pm$ std & {row['mean']:.4f} $\\pm$ {row['std']:.4f}"
            + r"\\"
        )
    real_rows.extend(
        [
            f"Training-median reference NMAE & {median['test_nmae']:.4f}" + r"\\",
            f"Mean paired fact Jaccard & {measure['mean_jaccard']:.4f}" + r"\\",
            f"Same-schema independent-seed drift & {verified['drift']['delta_seed_ind']:.4f}"
            + r"\\",
            f"Cross-view independent-seed drift & {verified['drift']['delta_view_ind']:.4f}"
            + r"\\",
            f"ESD [95\\% interval] & {verified['drift']['esd_ind']:.4f} [$ {verified['drift']['esd_ci_lo']:.4f},{verified['drift']['esd_ci_hi']:.4f}$]"
            + r"\\",
        ]
    )
    (target / "real_results.tex").write_text("\n".join(real_rows) + "\n")
    plt.rcParams.update(
        {
            "font.size": 9,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(6.3, 2.45), sharey=True, layout="constrained"
    )
    for ax, domain in zip(axes, DOMAINS, strict=True):
        for model, color, marker in zip(
            MODELS, ("#2166ac", "#b35806"), ("o", "s"), strict=True
        ):
            data = coverage[
                (coverage.domain == domain) & (coverage.model == model)
            ].sort_values("jaccard")
            ax.plot(
                data.jaccard,
                100 * data.paired_view_drift,
                marker=marker,
                ms=3,
                color=color,
                label=model,
            )
            ax.fill_between(
                data.jaccard,
                100 * data.paired_ci_lo,
                100 * data.paired_ci_hi,
                alpha=0.14,
                color=color,
            )
        ax.set_title("Synth-A" if domain == DOMAINS[0] else "Synth-B")
        ax.set_xlabel("Paired contribution-fact Jaccard")
        ax.set_xlim(0.15, 1.03)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Matched-seed disagreement (pp)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(ROOT / "paper/figures/coverage_ablation_v8.pdf")
    fig.savefig(ROOT / "paper/figures/coverage_ablation_v8.png", dpi=180)
    plt.close(fig)
    paper = ROOT / "paper/paper.tex"
    manuscript = paper.read_text()
    for filename in (
        "controlled_performance",
        "controlled_drift",
        "controlled_role",
        "real_results",
    ):
        begin, end = f"% BEGIN GENERATED {filename}", f"% END GENERATED {filename}"
        rendered = begin + "\n" + (target / f"{filename}.tex").read_text() + end
        directive = "\\input{generated/" + filename + ".tex}"
        if directive in manuscript:
            manuscript = manuscript.replace(directive, rendered)
        elif begin in manuscript and end in manuscript:
            left = manuscript.index(begin)
            right = manuscript.index(end, left) + len(end)
            manuscript = manuscript[:left] + rendered + manuscript[right:]
        else:
            raise ValueError(f"Missing generated table anchor: {filename}")
    paper.write_text(manuscript)
    print("Generated three controlled tables and paired-drift figure from v8 data")


if __name__ == "__main__":
    main()
