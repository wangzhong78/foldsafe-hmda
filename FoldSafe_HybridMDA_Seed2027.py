#!/usr/bin/env python3
"""Fold-safe hybrid extension of FoldSafe-TreeMDA (seed 2027).

The final score combines four independently auditable branches:

1. ExtraTrees on the original 133 fold-specific features (main branch);
2. histogram gradient boosting on the same features;
3. an NNSFMDA-inspired bounded nuclear-norm completion + attention branch;
4. a deliberately weakened RandomForest branch.

Fusion weights are selected only on an inner validation split. Outer-test labels
are never used to select a model, parameter, or weight. RandomForest is capped
at 0.10 and the ExtraTrees branch remains at least 0.65.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.extmath import randomized_svd


SEED = 2027
EPS = 1e-12


@dataclass(frozen=True)
class HybridConfig:
    seed: int = SEED
    folds: int = 5
    inner_fraction: float = 0.20
    components: int = 50
    extra_trees_estimators: int = 350
    random_forest_estimators: int = 120
    hgb_iterations: int = 250
    nnm_rank: int = 24
    nnm_iterations: int = 15
    nnm_lambda: float = 0.05
    completion_weight: float = 0.25
    attention_weight: float = 0.10
    attention_dim: int = 16
    weight_step: float = 0.05
    min_extra_weight: float = 0.65
    min_matrix_weight: float = 0.05
    max_matrix_weight: float = 0.10
    max_rf_weight: float = 0.10
    n_jobs: int = -1


def load_base_module(path: Path):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Base FoldSafe code not found: {path}")
    specification = importlib.util.spec_from_file_location("foldsafe_tree_base", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import base FoldSafe code: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def unit_scale(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    low = float(values.min())
    high = float(values.max())
    if high - low <= EPS:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - low) / (high - low)).astype(np.float32)


def soft_impute(
    interaction_microbe_drug: np.ndarray,
    initialization: np.ndarray,
    rank: int,
    iterations: int,
    shrinkage: float,
    seed: int,
) -> tuple[np.ndarray, list[float]]:
    """Bounded proximal SVT using training positives only.

    Every update is projected to [0, 1], which is the bounded constraint used
    by the NNSFMDA-style matrix-completion branch.
    """
    observed = interaction_microbe_drug > 0.5
    x = unit_scale(initialization).astype(np.float64)
    x[observed] = 1.0
    effective_rank = min(rank, min(x.shape) - 1)
    if effective_rank < 1:
        raise ValueError("NNM rank is invalid for the association matrix")
    changes: list[float] = []
    for iteration in range(iterations):
        filled = x.copy()
        filled[observed] = 1.0
        u, singular, vt = randomized_svd(
            filled,
            n_components=effective_rank,
            n_iter=4,
            random_state=seed + iteration,
        )
        singular = np.maximum(singular - shrinkage, 0.0)
        updated = np.clip((u * singular[None, :]) @ vt, 0.0, 1.0)
        updated[observed] = 1.0
        change = float(np.linalg.norm(updated - x) / max(np.linalg.norm(x), EPS))
        changes.append(change)
        x = updated
    return unit_scale(x), changes


def matrix_branch(
    base,
    interaction_drug_microbe: np.ndarray,
    microbe_external: np.ndarray,
    drug_external: np.ndarray,
    config: HybridConfig,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """NNSFMDA-inspired bounded completion and single-layer attention.

    The publication describes bounded nuclear-norm completion followed by a
    simplified global Transformer layer, but does not provide released code in
    this package. Here the completed association matrix is inserted back into
    the heterogeneous network; a truncated global encoder supplies query/key
    embeddings and a single cross-attention score. This is explicitly an
    auditable reconstruction, not a claim of author-code identity.
    """
    y = np.asarray(interaction_drug_microbe.T, dtype=np.float64)
    microbe_gip = base.gip_similarity(y)
    drug_gip = base.gip_similarity(y.T)
    microbe_fused = 0.5 * (microbe_gip + microbe_external)
    drug_fused = 0.5 * (drug_gip + drug_external)
    np.fill_diagonal(microbe_fused, 1.0)
    np.fill_diagonal(drug_fused, 1.0)

    # Similarity propagation initializes the bounded matrix-completion problem.
    # No held-out positive is present in y.
    left = base.row_normalize(microbe_fused) @ y
    right = y @ base.row_normalize(drug_fused).T
    bilinear = unit_scale(0.5 * (left + right))
    nnm, convergence = soft_impute(
        y,
        bilinear,
        config.nnm_rank,
        config.nnm_iterations,
        config.nnm_lambda,
        seed,
    )
    # Simplified one-layer global attention on the completed heterogeneous
    # network. SVD supplies deterministic low-rank Q/K projections; sigmoid is
    # used because pair scores are independent probabilities rather than a
    # categorical distribution over all microbes.
    heterogeneous = np.block(
        [[microbe_fused, nnm], [nnm.T, drug_fused]]
    ).astype(np.float64)
    transition = base.row_normalize(heterogeneous)
    attention_dim = min(config.attention_dim, transition.shape[0] - 1)
    embedding, singular, _ = randomized_svd(
        transition,
        n_components=attention_dim,
        n_iter=5,
        random_state=seed + 1000,
    )
    embedding = embedding * np.sqrt(np.maximum(singular, 0.0))[None, :]
    embedding -= embedding.mean(axis=1, keepdims=True)
    embedding /= np.maximum(embedding.std(axis=1, keepdims=True), EPS)
    n_microbes = y.shape[0]
    query = embedding[:n_microbes]
    key = embedding[n_microbes:]
    logits = (query @ key.T) / np.sqrt(max(attention_dim, 1))
    logits = np.clip(logits, -30.0, 30.0)
    attention = unit_scale(1.0 / (1.0 + np.exp(-logits)))

    # Propagation is retained as the stable residual, bounded completion adds
    # low-rank structure, and attention performs a lightweight final refinement.
    propagation_weight = 1.0 - config.completion_weight - config.attention_weight
    if propagation_weight < 0.0:
        raise ValueError("completion_weight + attention_weight must not exceed 1")
    fused = (
        propagation_weight * bilinear
        + config.completion_weight * nnm
        + config.attention_weight * attention
    )
    return unit_scale(fused), {
        "nnm_iterations": len(convergence),
        "nnm_final_relative_change": convergence[-1] if convergence else None,
        "bounded_projection": "[0,1]",
        "attention_dim": attention_dim,
        "attention_layers": 1,
    }


def score_pairs(score_matrix: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return score_matrix[pairs[:, 1], pairs[:, 0]]


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr": float(average_precision_score(labels, scores)),
    }


def extra_model(config: HybridConfig, seed: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=config.extra_trees_estimators,
        max_features=0.25,
        min_samples_leaf=1,
        n_jobs=config.n_jobs,
        random_state=seed,
    )


def hgb_model(config: HybridConfig, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=config.hgb_iterations,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=seed,
    )


def weak_rf_model(config: HybridConfig, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.random_forest_estimators,
        max_features="sqrt",
        max_depth=12,
        min_samples_leaf=2,
        n_jobs=config.n_jobs,
        random_state=seed,
    )


def predict_supervised(
    config: HybridConfig,
    train_x: np.ndarray,
    train_y: np.ndarray,
    target_x: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    models = {
        "extra": extra_model(config, seed + 1),
        "hgb": hgb_model(config, seed + 2),
        "rf": weak_rf_model(config, seed + 3),
    }
    scores: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        scores[name] = model.predict_proba(target_x)[:, 1]
    return scores, models


def weight_grid(config: HybridConfig) -> list[dict[str, float]]:
    step = config.weight_step
    count = int(round(1.0 / step))
    if step <= 0 or not np.isclose(count * step, 1.0):
        raise ValueError("weight_step must be positive and divide 1 exactly")
    values = np.arange(count + 1, dtype=float) * step
    rows: list[dict[str, float]] = []
    for matrix_weight in values:
        if matrix_weight < config.min_matrix_weight - EPS:
            continue
        if matrix_weight > config.max_matrix_weight + EPS:
            continue
        for hgb_weight in values:
            for rf_weight in values:
                extra_weight = 1.0 - matrix_weight - hgb_weight - rf_weight
                if extra_weight < config.min_extra_weight - EPS:
                    continue
                if rf_weight > config.max_rf_weight + EPS:
                    continue
                if extra_weight < -EPS:
                    continue
                rows.append(
                    {
                        "extra": float(extra_weight),
                        "hgb": float(hgb_weight),
                        "matrix": float(matrix_weight),
                        "rf": float(rf_weight),
                    }
                )
    return rows


def blend(scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return sum(weights[name] * scores[name] for name in ("extra", "hgb", "matrix", "rf"))


def choose_weights(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    config: HybridConfig,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Select on inner validation within the pre-specified hybrid constraints."""
    baseline_weights = {
        "extra": 1.0 - config.min_matrix_weight,
        "hgb": 0.0,
        "matrix": config.min_matrix_weight,
        "rf": 0.0,
    }
    baseline = metrics(labels, blend(scores, baseline_weights))
    rows: list[dict[str, float]] = []
    for weights in weight_grid(config):
        value = metrics(labels, blend(scores, weights))
        rows.append({**weights, **value})
    eligible = [
        row
        for row in rows
        if row["auroc"] + EPS >= baseline["auroc"]
        and row["aupr"] + EPS >= baseline["aupr"]
    ]
    if not eligible:
        return baseline_weights, rows
    selected = max(
        eligible,
        key=lambda row: (
            row["auroc"] + row["aupr"],
            row["aupr"],
            row["auroc"],
            -row["rf"],
            row["matrix"] + row["hgb"],
        ),
    )
    return {name: float(selected[name]) for name in ("extra", "hgb", "matrix", "rf")}, rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    fold_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
) -> dict[str, object]:
    labels = np.asarray([int(row["label"]) for row in prediction_rows])
    mapping = {
        "hybrid": "hybrid_score",
        "extra": "extra_score",
        "hgb": "hgb_score",
        "matrix": "matrix_score",
        "weak_rf": "rf_score",
    }
    summary: dict[str, object] = {}
    for name, column in mapping.items():
        aucs = np.asarray([float(row[f"{name}_auroc"]) for row in fold_rows])
        auprs = np.asarray([float(row[f"{name}_aupr"]) for row in fold_rows])
        pooled_score = np.asarray([float(row[column]) for row in prediction_rows])
        summary[name] = {
            "auroc": {
                "mean": float(aucs.mean()),
                "std": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
            },
            "aupr": {
                "mean": float(auprs.mean()),
                "std": float(auprs.std(ddof=1)) if len(auprs) > 1 else None,
            },
            "pooled": metrics(labels, pooled_score),
        }
    return summary


def plot_curves(prediction_rows: list[dict[str, object]], output: Path) -> None:
    labels = np.asarray([int(row["label"]) for row in prediction_rows])
    branches = {
        "Hybrid": np.asarray([float(row["hybrid_score"]) for row in prediction_rows]),
        "ExtraTrees": np.asarray([float(row["extra_score"]) for row in prediction_rows]),
        "NNSFMDA-inspired": np.asarray([float(row["matrix_score"]) for row in prediction_rows]),
    }
    colors = {"Hybrid": "#b2182b", "ExtraTrees": "#2166ac", "NNSFMDA-inspired": "#1b7837"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), dpi=180)
    for name, score in branches.items():
        fpr, tpr, _ = roc_curve(labels, score)
        precision, recall, _ = precision_recall_curve(labels, score)
        axes[0].plot(fpr, tpr, lw=2 if name == "Hybrid" else 1.3, color=colors[name], label=f"{name} ({roc_auc_score(labels, score):.4f})")
        axes[1].plot(recall, precision, lw=2 if name == "Hybrid" else 1.3, color=colors[name], label=f"{name} ({average_precision_score(labels, score):.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color="#999999", lw=1)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="Out-of-fold ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Out-of-fold precision–recall")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def run(
    base_code: Path,
    data_dir: Path,
    brmda_view: Path,
    output: Path,
    config: HybridConfig,
    max_folds: int | None,
) -> dict[str, object]:
    base = load_base_module(base_code)
    data = base.load_mdad(data_dir)
    microbe_external, drug_external = base.load_brmda_external(brmda_view, data)
    base_config = base.TreeConfig(
        seed=config.seed,
        folds=config.folds,
        inner_fraction=config.inner_fraction,
        tree_estimators=config.extra_trees_estimators,
        brmda_components=config.components,
        weight_step=0.005,
        n_jobs=config.n_jobs,
    )
    folds = base.make_folds(data, base_config)
    if max_folds is not None:
        folds = folds[:max_folds]
    output.mkdir(parents=True, exist_ok=True)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []
    started = time.time()

    for fold in folds:
        print(f"Fold {fold.number}/{config.folds}", flush=True)
        outer_interaction = data.association.copy()
        outer_interaction[fold.test_positive[:, 0], fold.test_positive[:, 1]] = 0.0
        if outer_interaction[fold.test_positive[:, 0], fold.test_positive[:, 1]].any():
            raise AssertionError("Outer test positives remain in the training graph")
        outer_pairs = np.vstack([fold.train_pairs, fold.test_pairs])
        outer_x, _ = base.build_compact_features(
            data,
            microbe_external,
            drug_external,
            outer_interaction,
            outer_pairs,
            config.components,
            41000 + fold.number,
        )
        outer_train_x = outer_x[: len(fold.train_pairs)]
        outer_test_x = outer_x[len(fold.train_pairs) :]

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.inner_fraction,
            random_state=config.seed + 42000 + fold.number,
        )
        inner_train_index, inner_val_index = next(
            splitter.split(fold.train_pairs, fold.train_labels)
        )
        inner_train_pairs = fold.train_pairs[inner_train_index]
        inner_train_labels = fold.train_labels[inner_train_index]
        inner_val_pairs = fold.train_pairs[inner_val_index]
        inner_val_labels = fold.train_labels[inner_val_index]

        inner_interaction = outer_interaction.copy()
        inner_val_positive = inner_val_pairs[inner_val_labels == 1]
        inner_interaction[inner_val_positive[:, 0], inner_val_positive[:, 1]] = 0.0
        if inner_interaction[inner_val_positive[:, 0], inner_val_positive[:, 1]].any():
            raise AssertionError("Inner validation positives remain in the training graph")
        inner_pairs = np.vstack([inner_train_pairs, inner_val_pairs])
        inner_x, _ = base.build_compact_features(
            data,
            microbe_external,
            drug_external,
            inner_interaction,
            inner_pairs,
            config.components,
            43000 + fold.number,
        )
        inner_train_x = inner_x[: len(inner_train_pairs)]
        inner_val_x = inner_x[len(inner_train_pairs) :]

        inner_scores, _ = predict_supervised(
            config,
            inner_train_x,
            inner_train_labels,
            inner_val_x,
            44000 + fold.number * 10,
        )
        inner_matrix, _ = matrix_branch(
            base,
            inner_interaction,
            microbe_external,
            drug_external,
            config,
            45000 + fold.number * 100,
        )
        inner_scores["matrix"] = score_pairs(inner_matrix, inner_val_pairs)
        weights, grid = choose_weights(inner_val_labels, inner_scores, config)
        for row in grid:
            weight_rows.append({"fold": fold.number, **row})

        outer_scores, _ = predict_supervised(
            config,
            outer_train_x,
            fold.train_labels,
            outer_test_x,
            46000 + fold.number * 10,
        )
        outer_matrix, matrix_diagnostics = matrix_branch(
            base,
            outer_interaction,
            microbe_external,
            drug_external,
            config,
            47000 + fold.number * 100,
        )
        outer_scores["matrix"] = score_pairs(outer_matrix, fold.test_pairs)
        hybrid_score = blend(outer_scores, weights)
        branch_scores = {
            "hybrid": hybrid_score,
            "extra": outer_scores["extra"],
            "hgb": outer_scores["hgb"],
            "matrix": outer_scores["matrix"],
            "weak_rf": outer_scores["rf"],
        }
        branch_metrics = {
            name: metrics(fold.test_labels, score)
            for name, score in branch_scores.items()
        }
        fold_row: dict[str, object] = {
            "fold": fold.number,
            **{f"weight_{name}": value for name, value in weights.items()},
            "train_positive": int(fold.train_labels.sum()),
            "train_negative": int(len(fold.train_labels) - fold.train_labels.sum()),
            "test_positive": int(fold.test_labels.sum()),
            "test_negative": int(len(fold.test_labels) - fold.test_labels.sum()),
            "held_out_positive_sum_in_training_A": float(
                outer_interaction[fold.test_positive[:, 0], fold.test_positive[:, 1]].sum()
            ),
            **matrix_diagnostics,
        }
        for name, value in branch_metrics.items():
            fold_row[f"{name}_auroc"] = value["auroc"]
            fold_row[f"{name}_aupr"] = value["aupr"]
        fold_rows.append(fold_row)
        for index, pair in enumerate(fold.test_pairs):
            prediction_rows.append(
                {
                    "fold": fold.number,
                    "drug": int(pair[0]),
                    "microbe": int(pair[1]),
                    "label": int(fold.test_labels[index]),
                    "hybrid_score": float(hybrid_score[index]),
                    "extra_score": float(outer_scores["extra"][index]),
                    "hgb_score": float(outer_scores["hgb"][index]),
                    "matrix_score": float(outer_scores["matrix"][index]),
                    "rf_score": float(outer_scores["rf"][index]),
                }
            )
        value = branch_metrics["hybrid"]
        print(
            f"  weights={weights}; AUROC/AUPR={value['auroc']:.6f}/{value['aupr']:.6f}",
            flush=True,
        )

    summary = summarize(fold_rows, prediction_rows)
    result = {
        "protocol": {
            "model": "FoldSafe-HybridMDA",
            "balanced_positive_to_pseudonegative_ratio": "1:1",
            "outer_positive_edges_masked_before_all_features_and_matrix_completion": True,
            "inner_positive_edges_masked_before_weight_selection": True,
            "outer_test_labels_used_for_weight_selection": False,
            "fusion": "inner-validation simplex grid",
            "random_forest_weight_cap": config.max_rf_weight,
            "extra_trees_minimum_weight": config.min_extra_weight,
            "nnsfmda_minimum_weight": config.min_matrix_weight,
            "nnsfmda_maximum_weight": config.max_matrix_weight,
            "matrix_branch": "NNSFMDA-inspired bounded nuclear norm completion plus single-layer global attention",
            "cv_scope": "pairwise transductive missing-link prediction",
        },
        "config": asdict(config),
        "folds_run": len(folds),
        "summary": summary,
        "fold_metrics": fold_rows,
        "runtime_seconds": time.time() - started,
    }
    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "predictions.csv", prediction_rows)
    write_csv(output / "inner_weight_grid.csv", weight_rows)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_curves(prediction_rows, output / "roc_pr_curves.png")
    print(json.dumps(summary["hybrid"], ensure_ascii=False, indent=2), flush=True)
    print(f"Results written to: {output.resolve()}", flush=True)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fold-safe HybridMDA, seed 2027")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--base-code",
        type=Path,
        default=script_dir / "FoldSafe_TreeMDA_Seed2027.py",
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--brmda-view", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("hybrid_results_seed2027"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = HybridConfig(folds=args.folds, n_jobs=args.n_jobs)
    try:
        run(
            args.base_code,
            args.data.resolve(),
            args.brmda_view.resolve(),
            args.output.resolve(),
            config,
            args.max_folds,
        )
    except (AssertionError, FileNotFoundError, FloatingPointError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
