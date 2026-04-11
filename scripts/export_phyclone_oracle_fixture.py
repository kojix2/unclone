#!/usr/bin/env python3
"""Export PhyClone oracle fixture JSON for Rust compat parity tests.

Usage example:
  python scripts/export_phyclone_oracle_fixture.py \
    --input-tsv examples/phy_oracle_input.tsv \
    --cluster-tsv examples/phy_oracle_clusters.tsv \
    --output rust-kernel/src/phyclone/compat/testdata/phyclone_oracle_fixture.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read_input_rows(path: Path, fallback_outlier_prob: float | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            outlier_prob = float(row["outlier_prob"]) if row.get("outlier_prob") else None
            if outlier_prob is None and fallback_outlier_prob is not None:
                outlier_prob = float(fallback_outlier_prob)

            rows.append(
                {
                    "mutation_id": row["mutation_id"],
                    "sample_id": row["sample_id"],
                    "ref_counts": int(row["ref_counts"]),
                    "alt_counts": int(row["alt_counts"]),
                    "major_cn": int(row["major_cn"]),
                    "minor_cn": int(row["minor_cn"]),
                    "normal_cn": int(row["normal_cn"]),
                    "tumour_content": float(row.get("tumour_content") or 1.0),
                    "error_rate": float(row.get("error_rate") or 1e-3),
                    "cluster_id": row.get("cluster_id") or None,
                    "outlier_prob": outlier_prob,
                }
            )
    return rows


def _read_cluster_rows(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "mutation_id": row["mutation_id"],
                    "cluster_id": row["cluster_id"],
                    "outlier_prob": float(row["outlier_prob"]) if row.get("outlier_prob") else None,
                }
            )
    return rows


def _build_cluster_size_map(cluster_rows: list[dict[str, object]]) -> dict[str, int]:
    if not cluster_rows:
        return {}

    members: dict[str, set[str]] = {}
    for row in cluster_rows:
        cluster_id = str(row["cluster_id"])
        mutation_id = str(row["mutation_id"])
        members.setdefault(cluster_id, set()).add(mutation_id)

    return {cluster_id: len(mutations) for cluster_id, mutations in members.items()}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-tsv", required=True, type=Path)
    parser.add_argument("--cluster-tsv", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--density", choices=["binomial", "beta-binomial"], default="beta-binomial")
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--precision", type=float, default=400.0)
    parser.add_argument("--outlier-prob", type=float, default=1e-4)
    parser.add_argument("--high-loss-prob", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--phyclone-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "PhyClone",
        help="Path to Roth-Lab/PhyClone repository root",
    )
    return parser


def _build_tree_prior_stats(tree) -> dict[str, object]:
    root_subtree_node_counts: list[int] = []
    for root in tree.roots:
        root_subtree_node_counts.append(int(tree.get_number_of_descendants(root) + 1))

    return {
        "num_nodes": int(tree.get_number_of_nodes()),
        "multiplicity_log": float(tree.multiplicity),
        "root_subtree_node_counts": root_subtree_node_counts,
    }


def _build_node_clusters(tree) -> list[dict[str, object]]:
    outlier_node_name = tree.outlier_node_name
    clusters: list[dict[str, object]] = []
    for node_id, data_list in tree.node_data.items():
        clusters.append(
            {
                "data_point_count": int(len(data_list)),
                "is_outlier_node": bool(node_id == outlier_node_name),
            }
        )
    return clusters


def _build_tree_likelihood(tree) -> dict[str, object]:
    return {
        "root_children_count": int(tree.get_number_of_children(tree.root_node_name)),
        "data_log_likelihood": tree.data_log_likelihood.tolist(),
    }


def _build_outlier_points(data_points) -> list[dict[str, object]]:
    return [
        {
            "id": int(dp.idx),
            "outlier_prob": float(dp.outlier_prob),
            "outlier_prob_not": float(dp.outlier_prob_not),
            "outlier_marginal_prob": float(dp.outlier_marginal_prob),
        }
        for dp in data_points
    ]


def _serialize_fixed_tree_case(name, tree, prior, joint, outlier_points) -> dict[str, object]:
    crp_prior, _ = prior.compute_CRP_prior(tree)
    log_p, log_p_one = joint.compute_log_p_and_log_p_one(tree)
    case = {
        "name": name,
        "tree_prior_stats": _build_tree_prior_stats(tree),
        "node_clusters": _build_node_clusters(tree),
        "likelihood": _build_tree_likelihood(tree),
        "outlier_points": outlier_points,
        "assigned_outlier_ids": [int(dp.idx) for dp in tree.outliers],
        "expected": {
            "crp_prior": float(crp_prior),
            "log_p": float(log_p),
            "log_p_one": float(log_p_one),
        },
    }
    return case


def _build_fixed_tree_oracle(data_points):
    from phyclone.tree.distributions import FSCRPDistribution, TreeJointDistribution
    from phyclone.tree.tree import Tree

    prior = FSCRPDistribution(alpha=1.0, c_const=1000.0)
    joint = TreeJointDistribution(prior, outlier_modelling_active=True)
    outlier_points = _build_outlier_points(data_points)

    cases: list[dict[str, object]] = []

    single = Tree.get_single_node_tree(data_points)
    cases.append(_serialize_fixed_tree_case("single_node_all_in_tree", single, prior, joint, outlier_points))

    if len(data_points) >= 2:
        one_outlier = single.copy()
        moved_dp = data_points[0]
        moved_node = one_outlier.labels[moved_dp.idx]
        one_outlier.remove_data_point_from_node(moved_dp, moved_node)
        one_outlier.add_data_point_to_outliers(moved_dp)
        cases.append(_serialize_fixed_tree_case("single_node_one_outlier", one_outlier, prior, joint, outlier_points))

        multi_root = Tree(data_points[0].grid_size)
        for dp in data_points:
            multi_root.create_root_node(data=[dp])
        cases.append(_serialize_fixed_tree_case("multi_root_each_datapoint", multi_root, prior, joint, outlier_points))

    return {
        "prior": {"alpha": 1.0, "c_const": 1000.0},
        "cases": cases,
    }


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    phyclone_root = args.phyclone_root.resolve()
    if not phyclone_root.exists():
        print(f"PhyClone root not found: {phyclone_root}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(phyclone_root))

    try:
        import numpy as np
        from phyclone.data.pyclone import load_data
    except Exception as exc:  # pragma: no cover
        print("Failed to import PhyClone stack. Install dependencies first.", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 3

    rng = np.random.default_rng(args.seed)

    data_points, _samples, _ = load_data(
        file_name=str(args.input_tsv),
        rng=rng,
        high_loss_prob=args.high_loss_prob,
        assign_loss_prob=False,
        user_provided_loss_prob=True,
        cluster_file=str(args.cluster_tsv) if args.cluster_tsv else None,
        density=args.density,
        grid_size=args.grid_size,
        outlier_prob=args.outlier_prob,
        precision=args.precision,
    )

    cluster_rows = _read_cluster_rows(args.cluster_tsv)
    cluster_size_map = _build_cluster_size_map(cluster_rows)

    payload = {
        "config": {
            "density": args.density,
            "precision": args.precision,
            "grid_size": args.grid_size,
        },
        "rows": _read_input_rows(
            args.input_tsv,
            None if args.cluster_tsv else args.outlier_prob,
        ),
        "cluster_rows": cluster_rows,
        "oracle": {
            "ccf_grid": np.linspace(0.0, 1.0, args.grid_size).tolist(),
            "datapoints": [
                {
                    "idx": int(dp.idx),
                    "name": str(dp.name),
                    "value": dp.value.tolist(),
                    "outlier_prob_log": float(dp.outlier_prob),
                    "outlier_prob_not_log": float(dp.outlier_prob_not),
                    "outlier_marginal_log": float(dp.outlier_marginal_prob),
                    "size": int(cluster_size_map.get(str(dp.name), 1)),
                }
                for dp in data_points
            ],
        },
        "fixed_tree_oracle": _build_fixed_tree_oracle(data_points),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote oracle fixture: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
