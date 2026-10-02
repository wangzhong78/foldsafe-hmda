#!/usr/bin/env python3
"""Reproducible FoldSafe-TreeMDA pipeline for the MDAD benchmark.

This file contains the complete primary tree-only pipeline:

1. Load the MDAD matrices and the BRMDA-inspired auxiliary similarities.
2. Sample a balanced set of pseudo-negatives and create pairwise CV folds.
3. Mask outer-test positive edges before recomputing every derived feature.
4. Mask inner-validation positive edges before model selection.
5. Build the 133-dimensional compact feature view with leave-one-out (LOO)
   corrections for observed candidate edges.
6. Select between RandomForest and a Pareto-dominant blend of two ExtraTrees
   models using inner validation only.
7. Refit the selected tree structure on the complete outer-training fold and
   report AUROC and average precision (AUPR/AP) on the untouched outer test.

No project-local Python modules and no PyTorch installation are required.
Dependencies: numpy and scikit-learn.

The experimental seed is intentionally fixed at 2027 throughout this release.

Example using a supplementary ZIP directly:

    python FoldSafe_TreeMDA_Seed2027.py \
        --source-zip FoldSafe-TreeMDA-Supplementary-Code.zip \
        --output tree_branch_results

Example using already extracted inputs:

    python FoldSafe_TreeMDA_Seed2027.py \
        --data path/to/data/MDAD \
        --brmda-view path/to/data/mdad_brmda_view.npz \
        --output tree_branch_results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

try:
    import numpy as np
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.utils.extmath import randomized_svd
except ImportError as exc:  # pragma: no cover - only used for a friendly CLI error
    raise SystemExit(
        "Missing dependency. Install with: python -m pip install numpy scikit-learn"
    ) from exc


REQUIRED_MDAD_FILES = (
    "drug_microbe_matrix.txt",
    "drug_structure_sim.txt",
    "microbe_function_sim.txt",
    "drug_drug_interactions.csv",
    "microbe_microbe_interactions.csv",
    "drug_with_disease.txt",
    "microbe_with_disease.txt",
)

EXPERIMENT_SEED = 2027


@dataclass(frozen=True)
class TreeConfig:
    seed: int = EXPERIMENT_SEED
    folds: int = 5
    inner_fraction: float = 0.20
    tree_estimators: int = 400
    brmda_components: int = 50
    weight_step: float = 0.005
    n_jobs: int = -1


@dataclass(frozen=True)
class MDAD:
    """All matrices used by the extracted tree branch."""

    association: np.ndarray  # drug x microbe
    drug_structure: np.ndarray
    microbe_function: np.ndarray
    drug_drug_edges: np.ndarray
    microbe_microbe_edges: np.ndarray
    drug_disease: np.ndarray
    microbe_disease: np.ndarray

    @property
    def n_drugs(self) -> int:
        return int(self.association.shape[0])

    @property
    def n_microbes(self) -> int:
        return int(self.association.shape[1])

    @property
    def n_diseases(self) -> int:
        return int(self.drug_disease.shape[1])


@dataclass(frozen=True)
class Fold:
    number: int
    train_pairs: np.ndarray  # drug, microbe
    train_labels: np.ndarray
    test_pairs: np.ndarray
    test_labels: np.ndarray
    test_positive: np.ndarray


@dataclass(frozen=True)
class BRMDAView:
    microbe_embedding: np.ndarray
    drug_embedding: np.ndarray
    microbe_propagation: np.ndarray
    drug_propagation: np.ndarray
    microbe_degree: np.ndarray
    drug_degree: np.ndarray

    def features(self, drug_microbe_pairs: np.ndarray) -> np.ndarray:
        pairs = np.asarray(drug_microbe_pairs, dtype=np.int64)
        drug_index, microbe_index = pairs[:, 0], pairs[:, 1]
        microbe = self.microbe_embedding[microbe_index]
        drug = self.drug_embedding[drug_index]
        product = microbe * drug
        difference = np.abs(microbe - drug)
        scalar = np.column_stack(
            [
                np.sum(product, axis=1),
                self.microbe_propagation[microbe_index, drug_index],
                self.drug_propagation[microbe_index, drug_index],
                self.microbe_degree[microbe_index],
                self.drug_degree[drug_index],
                self.microbe_degree[microbe_index] * self.drug_degree[drug_index],
            ]
        )
        return np.column_stack([product, difference, scalar]).astype(
            np.float32, copy=False
        )


def _edge_csv(path: Path, first: str, second: str) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray(
        [[int(row[first]) - 1, int(row[second]) - 1] for row in rows],
        dtype=np.int64,
    ).reshape(-1, 2)


def _one_based_edges(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if values.shape[1] != 2:
        raise ValueError(f"Expected two columns in {path}, got {values.shape}")
    return values - 1


def _edge_matrix(
    edges: np.ndarray, rows: int, columns: int, *, symmetric: bool = False
) -> np.ndarray:
    matrix = np.zeros((rows, columns), dtype=np.float32)
    if len(edges):
        matrix[edges[:, 0], edges[:, 1]] = 1.0
        if symmetric:
            matrix[edges[:, 1], edges[:, 0]] = 1.0
    return matrix


def _validate_edge_indices(edges: np.ndarray, rows: int, columns: int, name: str) -> None:
    if not len(edges):
        return
    if (
        edges[:, 0].min() < 0
        or edges[:, 1].min() < 0
        or edges[:, 0].max() >= rows
        or edges[:, 1].max() >= columns
    ):
        raise ValueError(f"{name} contains an out-of-range one-based identifier")


def load_mdad(path: Path) -> MDAD:
    """Load and structurally validate the MDAD input directory."""

    path = path.resolve()
    missing = [name for name in REQUIRED_MDAD_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"MDAD directory {path} is missing: {', '.join(missing)}"
        )

    association = np.loadtxt(
        path / "drug_microbe_matrix.txt", dtype=np.float32, ndmin=2
    )
    drug_structure = np.loadtxt(
        path / "drug_structure_sim.txt", dtype=np.float32, ndmin=2
    )
    microbe_function = np.loadtxt(
        path / "microbe_function_sim.txt", dtype=np.float32, ndmin=2
    )
    n_drugs, n_microbes = association.shape
    if drug_structure.shape != (n_drugs, n_drugs):
        raise ValueError(
            f"Drug similarity shape {drug_structure.shape} does not match "
            f"association shape {association.shape}"
        )
    if microbe_function.shape != (n_microbes, n_microbes):
        raise ValueError(
            f"Microbe similarity shape {microbe_function.shape} does not match "
            f"association shape {association.shape}"
        )
    if not np.all((association == 0) | (association == 1)):
        raise ValueError("drug_microbe_matrix.txt must be binary")

    drug_drug_edges = _edge_csv(
        path / "drug_drug_interactions.csv", "drug_id_1", "drug_id_2"
    )
    microbe_microbe_edges = _edge_csv(
        path / "microbe_microbe_interactions.csv", "microbe_id_1", "microbe_id_2"
    )
    drug_disease_edges = _one_based_edges(path / "drug_with_disease.txt")
    microbe_disease_edges = _one_based_edges(path / "microbe_with_disease.txt")
    n_diseases = max(
        int(drug_disease_edges[:, 1].max(initial=-1)),
        int(microbe_disease_edges[:, 1].max(initial=-1)),
    ) + 1
    if n_diseases <= 0:
        raise ValueError("No disease identifiers were found")

    _validate_edge_indices(drug_drug_edges, n_drugs, n_drugs, "drug-drug edges")
    _validate_edge_indices(
        microbe_microbe_edges, n_microbes, n_microbes, "microbe-microbe edges"
    )
    _validate_edge_indices(
        drug_disease_edges, n_drugs, n_diseases, "drug-disease edges"
    )
    _validate_edge_indices(
        microbe_disease_edges, n_microbes, n_diseases, "microbe-disease edges"
    )

    return MDAD(
        association=association,
        drug_structure=drug_structure,
        microbe_function=microbe_function,
        drug_drug_edges=drug_drug_edges,
        microbe_microbe_edges=microbe_microbe_edges,
        drug_disease=_edge_matrix(drug_disease_edges, n_drugs, n_diseases),
        microbe_disease=_edge_matrix(
            microbe_disease_edges, n_microbes, n_diseases
        ),
    )


def load_brmda_external(path: Path, data: MDAD) -> tuple[np.ndarray, np.ndarray]:
    """Load the external microbe/drug similarity view supplied with the project."""

    with np.load(path, allow_pickle=False) as values:
        required = {"association", "microbe_similarity", "drug_similarity"}
        missing = sorted(required.difference(values.files))
        if missing:
            raise ValueError(f"{path} is missing arrays: {', '.join(missing)}")
        association = values["association"].astype(np.float64)
        microbe = values["microbe_similarity"].astype(np.float64)
        drug = values["drug_similarity"].astype(np.float64)
    if not np.array_equal(association, data.association.T):
        raise ValueError(
            "The BRMDA auxiliary view and MDAD directory contain different "
            "association matrices"
        )
    if microbe.shape != (data.n_microbes, data.n_microbes):
        raise ValueError(f"Unexpected microbe similarity shape: {microbe.shape}")
    if drug.shape != (data.n_drugs, data.n_drugs):
        raise ValueError(f"Unexpected drug similarity shape: {drug.shape}")
    return microbe, drug


def _kfold_indices(
    n: int, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n, dtype=np.int64)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    sizes = np.full(folds, n // folds, dtype=np.int64)
    sizes[: n % folds] += 1
    result: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    all_indices = np.arange(n, dtype=np.int64)
    for size in sizes:
        stop = start + int(size)
        test = np.sort(indices[start:stop])
        keep = np.ones(n, dtype=bool)
        keep[test] = False
        result.append((all_indices[keep], test))
        start = stop
    return result


def make_folds(data: MDAD, config: TreeConfig) -> list[Fold]:
    """Create the balanced pairwise folds used by the audited experiment."""

    if config.folds < 2:
        raise ValueError("--folds must be at least 2")
    microbe_drug = data.association.T
    positives = np.argwhere(microbe_drug > 0).astype(np.int64)
    unlabeled = np.argwhere(microbe_drug == 0).astype(np.int64)
    if len(unlabeled) < len(positives):
        raise ValueError("Not enough unlabeled pairs for balanced negative sampling")
    if config.folds > len(positives):
        raise ValueError("--folds exceeds the number of positive pairs")

    rng = np.random.default_rng(config.seed)
    negatives = unlabeled[
        rng.choice(len(unlabeled), size=len(positives), replace=False)
    ]
    pos_splits = _kfold_indices(len(positives), config.folds, config.seed)
    neg_splits = _kfold_indices(len(negatives), config.folds, config.seed)
    folds: list[Fold] = []
    for number, ((pos_train, pos_test), (neg_train, neg_test)) in enumerate(
        zip(pos_splits, neg_splits, strict=True), start=1
    ):
        train_positive = positives[pos_train][:, ::-1].copy()
        train_negative = negatives[neg_train][:, ::-1].copy()
        test_positive = positives[pos_test][:, ::-1].copy()
        test_negative = negatives[neg_test][:, ::-1].copy()
        folds.append(
            Fold(
                number=number,
                train_pairs=np.vstack([train_positive, train_negative]),
                train_labels=np.r_[
                    np.ones(len(train_positive), dtype=np.int64),
                    np.zeros(len(train_negative), dtype=np.int64),
                ],
                test_pairs=np.vstack([test_positive, test_negative]),
                test_labels=np.r_[
                    np.ones(len(test_positive), dtype=np.int64),
                    np.zeros(len(test_negative), dtype=np.int64),
                ],
                test_positive=test_positive,
            )
        )
    return folds


def gip_similarity(profiles: np.ndarray) -> np.ndarray:
    values = np.asarray(profiles, dtype=np.float64)
    norm = np.einsum("ij,ij->i", values, values)
    gamma = len(values) / max(float(norm.sum()), 1e-12)
    distance = norm[:, None] + norm[None, :] - 2.0 * values @ values.T
    np.maximum(distance, 0.0, out=distance)
    result = np.exp(-gamma * distance)
    np.fill_diagonal(result, 1.0)
    return result


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    denominator = values.sum(axis=1, keepdims=True)
    return np.divide(
        values,
        denominator,
        out=np.zeros_like(values),
        where=denominator != 0,
    )


def build_brmda_view(
    interaction: np.ndarray,
    microbe_external: np.ndarray,
    drug_external: np.ndarray,
    components: int,
    seed: int,
) -> BRMDAView:
    y = np.asarray(interaction.T, dtype=np.float64)
    microbe_fused = 0.5 * (gip_similarity(y) + microbe_external)
    drug_fused = 0.5 * (gip_similarity(y.T) + drug_external)
    np.fill_diagonal(microbe_fused, 1.0)
    np.fill_diagonal(drug_fused, 1.0)
    microbe_propagation = row_normalize(microbe_fused) @ y
    drug_propagation = y @ row_normalize(drug_fused).T
    heterogeneous = np.block([[microbe_fused, y], [y.T, drug_fused]])
    transition = row_normalize(heterogeneous)
    k = min(components, transition.shape[0] - 1)
    if k < 1:
        raise ValueError("The heterogeneous matrix is too small for SVD features")
    left, singular, _ = randomized_svd(
        transition, n_components=k, n_iter=5, random_state=seed
    )
    embedding = left * np.sqrt(np.maximum(singular, 0.0))[None, :]
    microbe_degree = np.log1p(y.sum(axis=1))
    drug_degree = np.log1p(y.sum(axis=0))
    microbe_degree /= max(float(microbe_degree.max(initial=0.0)), 1e-12)
    drug_degree /= max(float(drug_degree.max(initial=0.0)), 1e-12)
    n_microbes = y.shape[0]
    return BRMDAView(
        microbe_embedding=embedding[:n_microbes],
        drug_embedding=embedding[n_microbes:],
        microbe_propagation=microbe_propagation,
        drug_propagation=drug_propagation,
        microbe_degree=microbe_degree,
        drug_degree=drug_degree,
    )


def symmetric_edge_matrix(edges: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros((size, size), dtype=np.float64)
    if len(edges):
        result[edges[:, 0], edges[:, 1]] = 1.0
        result[edges[:, 1], edges[:, 0]] = 1.0
    np.fill_diagonal(result, 1.0)
    return result


def disease_features(data: MDAD, pairs: np.ndarray) -> np.ndarray:
    drug_index = pairs[:, 0]
    microbe_index = pairs[:, 1]
    drug_disease = np.asarray(data.drug_disease, dtype=np.float64)
    microbe_disease = np.asarray(data.microbe_disease, dtype=np.float64)
    idf = (
        np.log(
            (1.0 + drug_disease.shape[0] + microbe_disease.shape[0])
            / (
                1.0
                + drug_disease.sum(axis=0)
                + microbe_disease.sum(axis=0)
            )
        )
        + 1.0
    )
    common = np.sum(
        drug_disease[drug_index] * microbe_disease[microbe_index], axis=1
    )
    weighted_common = np.sum(
        drug_disease[drug_index]
        * microbe_disease[microbe_index]
        * idf[None, :],
        axis=1,
    )
    drug_degree = drug_disease[drug_index].sum(axis=1)
    microbe_degree = microbe_disease[microbe_index].sum(axis=1)
    union = drug_degree + microbe_degree - common
    jaccard = np.divide(
        common, union, out=np.zeros_like(common), where=union > 0
    )
    cosine = np.divide(
        common,
        np.sqrt(drug_degree * microbe_degree),
        out=np.zeros_like(common),
        where=(drug_degree * microbe_degree) > 0,
    )
    return np.column_stack(
        [
            common,
            weighted_common,
            jaccard,
            cosine,
            drug_degree,
            microbe_degree,
            drug_degree * microbe_degree,
        ]
    )


def neighbor_stats(
    microbe_similarity: np.ndarray,
    drug_similarity: np.ndarray,
    y: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    rows: list[tuple[float, ...]] = []

    def stats(values: np.ndarray) -> tuple[float, float, float, float]:
        if not len(values):
            return 0.0, 0.0, 0.0, 0.0
        top = np.sort(values)[-min(3, len(values)) :]
        return (
            float(values.max()),
            float(top.mean()),
            float(values.mean()),
            float(values.sum()),
        )

    for drug, microbe in pairs:
        known_microbes = np.flatnonzero(y[:, drug] > 0)
        known_drugs = np.flatnonzero(y[microbe, :] > 0)
        # Leave the candidate itself out for observed training positives.
        known_microbes = known_microbes[known_microbes != microbe]
        known_drugs = known_drugs[known_drugs != drug]
        microbe_values = (
            microbe_similarity[microbe, known_microbes]
            if len(known_microbes)
            else np.empty(0)
        )
        drug_values = (
            drug_similarity[drug, known_drugs]
            if len(known_drugs)
            else np.empty(0)
        )
        rows.append((*stats(microbe_values), *stats(drug_values)))
    return np.asarray(rows, dtype=np.float64)


def matrix_scores(matrix: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    return matrix[pairs[:, 1], pairs[:, 0]]


def feature_names(components: int) -> list[str]:
    base = (
        [f"embedding_product_{index:02d}" for index in range(components)]
        + [f"embedding_absdiff_{index:02d}" for index in range(components)]
        + [
            "embedding_dot",
            "base_microbe_propagation_loo",
            "base_drug_propagation_loo",
            "microbe_degree",
            "drug_degree",
            "degree_product",
        ]
    )
    propagation = [
        f"{kind}_{side}_propagation_loo"
        for kind in ("external", "gip", "fused")
        for side in ("microbe", "drug")
    ] + [
        "dual_propagation_loo",
        "microbe_two_hop",
        "drug_two_hop",
        "mmi_path_loo",
        "ddi_path_loo",
        "mmi_ddi_path_loo",
    ]
    disease = [
        "disease_common",
        "disease_idf_common",
        "disease_jaccard",
        "disease_cosine",
        "drug_disease_degree",
        "microbe_disease_degree",
        "disease_degree_product",
    ]
    neighbor = [
        "microbe_neighbor_max",
        "microbe_neighbor_top3",
        "microbe_neighbor_mean",
        "microbe_neighbor_sum",
        "drug_neighbor_max",
        "drug_neighbor_top3",
        "drug_neighbor_mean",
        "drug_neighbor_sum",
    ]
    return base + propagation + disease + neighbor


def build_compact_features(
    data: MDAD,
    microbe_external: np.ndarray,
    drug_external: np.ndarray,
    interaction: np.ndarray,
    pairs: np.ndarray,
    components: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    """Build the fold-specific compact tree feature view.

    With the manuscript setting ``components=50``, the output has 133 columns:
    106 base variables, 12 propagation/path variables, 7 disease variables,
    and 8 neighborhood variables.
    """

    y = np.asarray(interaction.T, dtype=np.float64)
    microbe_gip = gip_similarity(y)
    drug_gip = gip_similarity(y.T)
    microbe_external = np.asarray(microbe_external, dtype=np.float64)
    drug_external = np.asarray(drug_external, dtype=np.float64)
    microbe_fused = 0.5 * (microbe_gip + microbe_external)
    drug_fused = 0.5 * (drug_gip + drug_external)
    np.fill_diagonal(microbe_fused, 1.0)
    np.fill_diagonal(drug_fused, 1.0)

    normalized_microbe = {
        "external": row_normalize(microbe_external),
        "gip": row_normalize(microbe_gip),
        "fused": row_normalize(microbe_fused),
    }
    normalized_drug = {
        "external": row_normalize(drug_external),
        "gip": row_normalize(drug_gip),
        "fused": row_normalize(drug_fused),
    }

    base_view = build_brmda_view(
        interaction,
        microbe_external,
        drug_external,
        components,
        seed,
    )
    base = base_view.features(pairs).astype(np.float64)
    actual_components = base_view.microbe_embedding.shape[1]
    observed = y[pairs[:, 1], pairs[:, 0]]
    # Columns 2*k+1 and 2*k+2 are the two base propagation scores.
    base[:, 2 * actual_components + 1] -= (
        observed
        * np.diag(normalized_microbe["fused"])[pairs[:, 1]]
    )
    base[:, 2 * actual_components + 2] -= (
        observed * np.diag(normalized_drug["fused"])[pairs[:, 0]]
    )

    scalar_columns: list[np.ndarray] = []
    for key in ("external", "gip", "fused"):
        left = normalized_microbe[key] @ y
        right = y @ normalized_drug[key].T
        left_score = matrix_scores(left, pairs) - (
            observed * np.diag(normalized_microbe[key])[pairs[:, 1]]
        )
        right_score = matrix_scores(right, pairs) - (
            observed * np.diag(normalized_drug[key])[pairs[:, 0]]
        )
        scalar_columns.extend([left_score, right_score])

    both = normalized_microbe["fused"] @ y @ normalized_drug["fused"].T
    both_score = matrix_scores(both, pairs) - (
        observed
        * np.diag(normalized_microbe["fused"])[pairs[:, 1]]
        * np.diag(normalized_drug["fused"])[pairs[:, 0]]
    )
    microbe_two_hop = (
        normalized_microbe["fused"] @ normalized_microbe["fused"] @ y
    )
    drug_two_hop = (
        y @ normalized_drug["fused"].T @ normalized_drug["fused"].T
    )
    scalar_columns.extend(
        [
            both_score,
            matrix_scores(microbe_two_hop, pairs),
            matrix_scores(drug_two_hop, pairs),
        ]
    )

    mmi = row_normalize(
        symmetric_edge_matrix(data.microbe_microbe_edges, data.n_microbes)
    )
    ddi = row_normalize(
        symmetric_edge_matrix(data.drug_drug_edges, data.n_drugs)
    )
    scalar_columns.extend(
        [
            matrix_scores(mmi @ y, pairs)
            - observed * np.diag(mmi)[pairs[:, 1]],
            matrix_scores(y @ ddi.T, pairs)
            - observed * np.diag(ddi)[pairs[:, 0]],
            matrix_scores(mmi @ y @ ddi.T, pairs)
            - observed
            * np.diag(mmi)[pairs[:, 1]]
            * np.diag(ddi)[pairs[:, 0]],
        ]
    )

    scalar = np.column_stack(scalar_columns)
    disease = disease_features(data, pairs)
    neighbor = neighbor_stats(
        microbe_fused, drug_fused, y, pairs
    )
    result = np.column_stack([base, scalar, disease, neighbor]).astype(np.float32)
    names = feature_names(actual_components)
    if result.shape[1] != len(names):
        raise AssertionError(
            f"Feature name count {len(names)} != matrix width {result.shape[1]}"
        )
    if not np.isfinite(result).all():
        raise FloatingPointError("Non-finite value found in the tree feature matrix")
    return result, names


def metrics(labels: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, score)),
        "aupr": float(average_precision_score(labels, score)),
    }


def select_pareto_weight(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    step: float,
) -> tuple[float, list[dict[str, float]]]:
    """Blend scores while retaining ``second`` unless both metrics are no worse."""

    steps = int(round(1.0 / step))
    if step <= 0 or not np.isclose(steps * step, 1.0):
        raise ValueError("--weight-step must be positive and divide 1 exactly")
    weights = np.linspace(0.0, 1.0, steps + 1)
    grid: list[dict[str, float]] = []
    for weight in weights:
        value = metrics(labels, (1.0 - weight) * first + weight * second)
        grid.append({"weight_second": float(weight), **value})
    baseline_index = len(grid) - 1
    reference = grid[baseline_index]
    eligible = [
        (index, row)
        for index, row in enumerate(grid)
        if row["auroc"] >= reference["auroc"]
        and row["aupr"] >= reference["aupr"]
    ]
    best_index, _ = max(
        eligible,
        key=lambda item: (
            item[1]["auroc"] + item[1]["aupr"],
            item[1]["aupr"],
            item[1]["auroc"],
            -abs(item[0] - baseline_index),
        ),
    )
    return float(weights[best_index]), grid


def extra_trees(
    seed: int, max_features: str | float, config: TreeConfig
) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=config.tree_estimators,
        max_features=max_features,
        min_samples_leaf=1,
        n_jobs=config.n_jobs,
        random_state=seed,
    )


def random_forest(seed: int, config: TreeConfig) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=config.tree_estimators,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=config.n_jobs,
        random_state=seed,
    )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summarize(
    fold_rows: Sequence[dict[str, object]],
    prediction_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    labels = np.asarray([int(row["label"]) for row in prediction_rows])
    mapping = {
        "random_forest": "rf_score",
        "extra_trees_025": "extra_025_score",
        "nested_tree": "nested_tree_score",
    }
    for name, column in mapping.items():
        score = np.asarray([float(row[column]) for row in prediction_rows])
        aucs = np.asarray(
            [float(row[f"{name}_auroc"]) for row in fold_rows], dtype=float
        )
        auprs = np.asarray(
            [float(row[f"{name}_aupr"]) for row in fold_rows], dtype=float
        )
        result[name] = {
            "auroc": {
                "mean": float(aucs.mean()),
                "std": float(aucs.std(ddof=1)) if len(aucs) > 1 else None,
            },
            "aupr": {
                "mean": float(auprs.mean()),
                "std": float(auprs.std(ddof=1)) if len(auprs) > 1 else None,
            },
            "pooled": metrics(labels, score),
        }
    return result


def run_tree_branch(
    data_dir: Path,
    brmda_view: Path,
    output: Path,
    config: TreeConfig,
    max_folds: int | None = None,
) -> dict[str, object]:
    """Run the complete fold-specific tree-ensemble cross-validation pipeline."""

    if not 0.0 < config.inner_fraction < 1.0:
        raise ValueError("--inner-fraction must be between 0 and 1")
    if config.tree_estimators < 1:
        raise ValueError("--trees must be positive")
    if config.brmda_components < 1:
        raise ValueError("--components must be positive")

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_mdad(data_dir)
    microbe_external, drug_external = load_brmda_external(brmda_view, data)
    folds = make_folds(data, config)
    if max_folds is not None:
        if not 1 <= max_folds <= len(folds):
            raise ValueError(f"--max-folds must be between 1 and {len(folds)}")
        folds = folds[:max_folds]

    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selected_importances: list[np.ndarray] = []
    names: list[str] | None = None
    started = time.time()

    for fold in folds:
        print(f"Fold {fold.number}/{config.folds}: outer fold-safe features", flush=True)
        outer_interaction = data.association.copy()
        outer_interaction[
            fold.test_positive[:, 0], fold.test_positive[:, 1]
        ] = 0.0
        if outer_interaction[
            fold.test_positive[:, 0], fold.test_positive[:, 1]
        ].any():
            raise AssertionError("Outer-test positive edges were not fully masked")
        outer_pairs = np.vstack([fold.train_pairs, fold.test_pairs])
        outer_all, outer_names = build_compact_features(
            data,
            microbe_external,
            drug_external,
            outer_interaction,
            outer_pairs,
            config.brmda_components,
            21000 + fold.number,
        )
        names = outer_names
        n_outer_train = len(fold.train_pairs)
        outer_train_x = outer_all[:n_outer_train]
        outer_test_x = outer_all[n_outer_train:]

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=config.inner_fraction,
            random_state=config.seed + 22000 + fold.number,
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
        inner_interaction[
            inner_val_positive[:, 0], inner_val_positive[:, 1]
        ] = 0.0
        if inner_interaction[
            inner_val_positive[:, 0], inner_val_positive[:, 1]
        ].any():
            raise AssertionError(
                "Inner-validation positive edges were not fully masked"
            )
        inner_pairs = np.vstack([inner_train_pairs, inner_val_pairs])
        inner_all, _ = build_compact_features(
            data,
            microbe_external,
            drug_external,
            inner_interaction,
            inner_pairs,
            config.brmda_components,
            23000 + fold.number,
        )
        n_inner_train = len(inner_train_pairs)
        inner_train_x = inner_all[:n_inner_train]
        inner_val_x = inner_all[n_inner_train:]

        inner_sqrt = extra_trees(24000 + fold.number, "sqrt", config).fit(
            inner_train_x, inner_train_labels
        )
        inner_025 = extra_trees(25000 + fold.number, 0.25, config).fit(
            inner_train_x, inner_train_labels
        )
        inner_rf = random_forest(26000 + fold.number, config).fit(
            inner_train_x, inner_train_labels
        )
        inner_sqrt_score = inner_sqrt.predict_proba(inner_val_x)[:, 1]
        inner_025_score = inner_025.predict_proba(inner_val_x)[:, 1]
        inner_rf_score = inner_rf.predict_proba(inner_val_x)[:, 1]

        nested_025_weight, _ = select_pareto_weight(
            inner_val_labels,
            inner_sqrt_score,
            inner_025_score,
            config.weight_step,
        )
        inner_extra_score = (
            (1.0 - nested_025_weight) * inner_sqrt_score
            + nested_025_weight * inner_025_score
        )
        extra_inner = metrics(inner_val_labels, inner_extra_score)
        rf_inner = metrics(inner_val_labels, inner_rf_score)
        nested_classifier = (
            "random_forest"
            if rf_inner["auroc"] >= extra_inner["auroc"]
            and rf_inner["aupr"] >= extra_inner["aupr"]
            else "extra_trees"
        )

        outer_sqrt = extra_trees(28000 + fold.number, "sqrt", config).fit(
            outer_train_x, fold.train_labels
        )
        outer_025 = extra_trees(29000 + fold.number, 0.25, config).fit(
            outer_train_x, fold.train_labels
        )
        outer_rf = random_forest(30000 + fold.number, config).fit(
            outer_train_x, fold.train_labels
        )
        sqrt_score = outer_sqrt.predict_proba(outer_test_x)[:, 1]
        extra_025_score = outer_025.predict_proba(outer_test_x)[:, 1]
        rf_score = outer_rf.predict_proba(outer_test_x)[:, 1]
        nested_extra_score = (
            (1.0 - nested_025_weight) * sqrt_score
            + nested_025_weight * extra_025_score
        )
        nested_tree_score = (
            rf_score
            if nested_classifier == "random_forest"
            else nested_extra_score
        )

        if nested_classifier == "random_forest":
            selected_importance = outer_rf.feature_importances_
        else:
            selected_importance = (
                (1.0 - nested_025_weight) * outer_sqrt.feature_importances_
                + nested_025_weight * outer_025.feature_importances_
            )
        selected_importances.append(np.asarray(selected_importance, dtype=float))

        rf_metric = metrics(fold.test_labels, rf_score)
        extra_025_metric = metrics(fold.test_labels, extra_025_score)
        nested_metric = metrics(fold.test_labels, nested_tree_score)
        fold_rows.append(
            {
                "fold": fold.number,
                "random_forest_auroc": rf_metric["auroc"],
                "random_forest_aupr": rf_metric["aupr"],
                "extra_trees_025_auroc": extra_025_metric["auroc"],
                "extra_trees_025_aupr": extra_025_metric["aupr"],
                "nested_tree_auroc": nested_metric["auroc"],
                "nested_tree_aupr": nested_metric["aupr"],
                "nested_classifier": nested_classifier,
                "nested_extra025_weight": nested_025_weight,
                "inner_random_forest_auroc": rf_inner["auroc"],
                "inner_random_forest_aupr": rf_inner["aupr"],
                "inner_extra_trees_auroc": extra_inner["auroc"],
                "inner_extra_trees_aupr": extra_inner["aupr"],
                "train_positive": int(fold.train_labels.sum()),
                "train_negative": int(
                    len(fold.train_labels) - fold.train_labels.sum()
                ),
                "test_positive": int(fold.test_labels.sum()),
                "test_negative": int(
                    len(fold.test_labels) - fold.test_labels.sum()
                ),
            }
        )
        for row_index, pair in enumerate(fold.test_pairs):
            prediction_rows.append(
                {
                    "fold": fold.number,
                    "row": row_index,
                    "drug": int(pair[0]),
                    "microbe": int(pair[1]),
                    "label": int(fold.test_labels[row_index]),
                    "rf_score": float(rf_score[row_index]),
                    "extra_sqrt_score": float(sqrt_score[row_index]),
                    "extra_025_score": float(extra_025_score[row_index]),
                    "nested_tree_score": float(nested_tree_score[row_index]),
                    "nested_classifier": nested_classifier,
                    "nested_extra025_weight": nested_025_weight,
                }
            )
        print(
            f"  selected={nested_classifier}, extra_025_weight="
            f"{nested_025_weight:.3f}, outer AUROC/AP="
            f"{nested_metric['auroc']:.6f}/{nested_metric['aupr']:.6f}",
            flush=True,
        )

    summary = _summarize(fold_rows, prediction_rows)
    assert names is not None
    importance_matrix = np.vstack(selected_importances)
    importance_mean = importance_matrix.mean(axis=0)
    importance_std = (
        importance_matrix.std(axis=0, ddof=1)
        if len(importance_matrix) > 1
        else np.zeros(importance_matrix.shape[1])
    )
    order = np.argsort(-importance_mean)
    importance_rows = [
        {
            "rank": rank,
            "feature": names[index],
            "mean_importance": float(importance_mean[index]),
            "std_importance": float(importance_std[index]),
        }
        for rank, index in enumerate(order, start=1)
    ]

    input_files = [data_dir / name for name in REQUIRED_MDAD_FILES]
    input_files.append(brmda_view)
    result = {
        "protocol": {
            "model": "FoldSafe-TreeMDA",
            "balanced_positive_to_pseudonegative_ratio": "1:1",
            "outer_positive_edges_masked_before_features": True,
            "inner_positive_edges_masked_before_selection": True,
            "candidate_self_contribution_removed_from_propagation_features": True,
            "outer_test_labels_used_for_model_selection": False,
            "cv_scope": "pairwise transductive missing-link prediction",
            "aupr_definition": "average precision",
            "feature_count": len(names),
            "provenance": "Standalone implementation supplied with the manuscript",
        },
        "config": asdict(config),
        "folds_run": len(folds),
        "summary": summary,
        "fold_metrics": fold_rows,
        "runtime": {
            "python": ".".join(str(value) for value in sys.version_info[:3]),
            "numpy": np.__version__,
            "elapsed_seconds": time.time() - started,
        },
        "input_manifest": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in input_files
        },
    }
    _write_csv(output / "fold_metrics.csv", fold_rows)
    _write_csv(output / "predictions.csv", prediction_rows)
    _write_csv(output / "feature_importance.csv", importance_rows)
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    nested = summary["nested_tree"]
    print(
        "Nested tree mean AUROC/AP: "
        f"{nested['auroc']['mean']:.6f}/{nested['aupr']['mean']:.6f}",
        flush=True,
    )
    print(f"Results written to: {output}", flush=True)
    return result


def _member_ending(
    archive: zipfile.ZipFile, ending: str
) -> zipfile.ZipInfo:
    normalized_ending = ending.replace("\\", "/")
    matches = [
        item
        for item in archive.infolist()
        if not item.is_dir()
        and item.filename.replace("\\", "/").endswith(normalized_ending)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one ZIP member ending in {ending!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _extract_required_inputs(
    source_zip: Path, destination: Path
) -> tuple[Path, Path]:
    """Extract only the required data files, using fixed safe destination names."""

    data_dir = destination / "data" / "MDAD"
    data_dir.mkdir(parents=True, exist_ok=True)
    view_path = destination / "data" / "mdad_brmda_view.npz"
    with zipfile.ZipFile(source_zip) as archive:
        for name in REQUIRED_MDAD_FILES:
            member = _member_ending(archive, f"/data/MDAD/{name}")
            with archive.open(member) as source, (data_dir / name).open("wb") as target:
                shutil.copyfileobj(source, target)
        member = _member_ending(archive, "/data/mdad_brmda_view.npz")
        view_path.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, view_path.open("wb") as target:
            shutil.copyfileobj(source, target)
    return data_dir, view_path


def _discover_inputs() -> tuple[Path, Path] | None:
    anchors = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    candidates: list[Path] = []
    for anchor in anchors:
        candidates.extend([anchor, *list(anchor.parents)[:4]])
    seen: set[Path] = set()
    for root in candidates:
        if root in seen:
            continue
        seen.add(root)
        direct_data = root / "data" / "MDAD"
        direct_view = root / "data" / "mdad_brmda_view.npz"
        if direct_data.is_dir() and direct_view.is_file():
            return direct_data, direct_view
        if root.is_dir():
            for data_dir in sorted(
                root.glob("FoldSafe-TreeMDA-Supplementary-Code-*/data/MDAD")
            ):
                view = data_dir.parent / "mdad_brmda_view.npz"
                if view.is_file():
                    return data_dir, view
    return None


@contextmanager
def resolve_inputs(
    data: Path | None,
    brmda_view: Path | None,
    source_zip: Path | None,
) -> Iterator[tuple[Path, Path]]:
    if source_zip is not None:
        if data is not None or brmda_view is not None:
            raise ValueError(
                "Use either --source-zip or the --data/--brmda-view pair, not both"
            )
        source_zip = source_zip.resolve()
        if not source_zip.is_file():
            raise FileNotFoundError(f"ZIP file not found: {source_zip}")
        with tempfile.TemporaryDirectory(prefix="foldsafe_tree_inputs_") as temp:
            yield _extract_required_inputs(source_zip, Path(temp))
        return

    if (data is None) != (brmda_view is None):
        raise ValueError("--data and --brmda-view must be supplied together")
    if data is not None and brmda_view is not None:
        yield data.resolve(), brmda_view.resolve()
        return

    discovered = _discover_inputs()
    if discovered is None:
        raise FileNotFoundError(
            "Could not locate the inputs automatically. Supply either "
            "--source-zip ZIP_FILE or both --data MDAD_DIR and "
            "--brmda-view NPZ_FILE."
        )
    yield discovered


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run FoldSafe-TreeMDA with the fixed experimental seed 2027"
        )
    )
    parser.add_argument(
        "--source-zip",
        type=Path,
        help="Original supplementary-code ZIP; required inputs are extracted temporarily",
    )
    parser.add_argument(
        "--data", type=Path, help="Extracted data/MDAD directory"
    )
    parser.add_argument(
        "--brmda-view", type=Path, help="Extracted data/mdad_brmda_view.npz"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tree_branch_results")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-fraction", type=float, default=0.20)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--components", type=int, default=50)
    parser.add_argument("--weight-step", type=float, default=0.005)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--max-folds",
        type=int,
        help="Run only the first N outer folds (for a quick smoke test)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = TreeConfig(
        seed=EXPERIMENT_SEED,
        folds=args.folds,
        inner_fraction=args.inner_fraction,
        tree_estimators=args.trees,
        brmda_components=args.components,
        weight_step=args.weight_step,
        n_jobs=args.n_jobs,
    )
    try:
        with resolve_inputs(args.data, args.brmda_view, args.source_zip) as (
            data_dir,
            brmda_view,
        ):
            run_tree_branch(
                data_dir,
                brmda_view,
                args.output,
                config,
                args.max_folds,
            )
    except (AssertionError, FileNotFoundError, FloatingPointError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
