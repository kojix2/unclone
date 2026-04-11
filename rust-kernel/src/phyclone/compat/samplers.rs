//! PhyClone-compatible MCMC move samplers.
//!
//! - `CompatDataPointSampler`  – mirrors `DataPointSampler` in `phyclone/mcmc/gibbs_mh.py`
//! - `CompatPruneRegraphSampler` – mirrors `PruneRegraphSampler` in `phyclone/mcmc/gibbs_mh.py`
//! - `CompatConcentrationSampler` – mirrors `GammaPriorConcentrationSampler` in
//!   `phyclone/mcmc/concentration.py`

#![allow(dead_code)]

use std::collections::HashSet;

use rand::{Rng, RngExt};

use super::data::CompatDataPoint;
use super::distributions::CompatTreeJointDistribution;
use super::root_permutation::fisher_yates_shuffle_pub;
use super::smc::sample_discrete_from_log_weights;
use super::tree_ids::DataPointId;
use super::tree_model::CompatTree;
use super::tree_stats::CompatOutlierPoint;

// ── DataPointSampler ──────────────────────────────────────────────────────────

/// Gibbs-MH sampler that re-assigns each data point to a new node.
///
/// Mirrors `DataPointSampler.sample_tree` in `phyclone/mcmc/gibbs_mh.py`:
/// 1. Iterate over all data point indices in a random order.
/// 2. For each data point whose current node contains >1 data points, propose
///    re-assigning it to every possible node (and optionally to the outlier set).
/// 3. Sample the new assignment proportional to `exp(log_p_one)`.
pub struct CompatDataPointSampler {
    pub joint: CompatTreeJointDistribution,
    pub outliers: bool,
}

impl CompatDataPointSampler {
    pub fn sample_tree(
        &self,
        data_points: &[CompatDataPoint],
        tree: &CompatTree,
        rng: &mut impl Rng,
    ) -> Result<CompatTree, String> {
        let outlier_points = CompatTree::outlier_points(data_points);

        // Collect non-outlier dp ids and shuffle.
        let mut dp_ids: Vec<DataPointId> = data_points.iter().map(|dp| dp.idx).collect();
        fisher_yates_shuffle_pub(&mut dp_ids, rng);

        let mut current_tree = tree.clone();

        for dp_id in dp_ids {
            // Determine current node.
            let current_node = current_tree
                .node_id_for_data_point(dp_id)
                .map(|s| s.to_string());

            // Determine whether to attempt a move:
            // - Outlier ("-1"): match PhyClone behaviour and only move when the outlier
            //   bucket contains >1 data points.
            // - Tree node: only move if the node currently contains >1 data points,
            //   because pulling the last data point out of a singleton node would
            //   leave an empty (dangling) node.
            let is_outlier = matches!(&current_node, Some(id) if id == "-1");
            if is_outlier {
                if current_tree.assigned_outlier_ids().len() <= 1 {
                    continue;
                }
            } else {
                let node_size = current_node
                    .as_deref()
                    .and_then(|id| current_tree.node(id))
                    .map(|n| n.data_point_ids.len())
                    .unwrap_or(0);
                if node_size <= 1 {
                    continue;
                }
            }

            // Find the data point value from data_points slice.
            let dp = match data_points.iter().find(|d| d.idx == dp_id) {
                Some(d) => d,
                None => continue,
            };

            // Build candidate trees by assigning dp to every non-root node
            // (and optionally to outliers).
            let candidate_trees = build_dp_candidate_trees(
                &current_tree,
                dp,
                current_node.as_deref(),
                self.outliers,
                &outlier_points,
                &self.joint,
            )?;

            if candidate_trees.is_empty() {
                continue;
            }

            let log_scores: Vec<f64> = candidate_trees.iter().map(|(score, _)| *score).collect();

            let chosen = sample_discrete_from_log_weights(&log_scores, rng)?;
            current_tree = candidate_trees.into_iter().nth(chosen).unwrap().1;
        }

        Ok(current_tree)
    }
}

fn build_dp_candidate_trees(
    tree: &CompatTree,
    dp: &CompatDataPoint,
    current_node_id: Option<&str>,
    outliers_active: bool,
    outlier_points: &[CompatOutlierPoint],
    joint: &CompatTreeJointDistribution,
) -> Result<Vec<(f64, CompatTree)>, String> {
    let mut candidates = Vec::new();

    // Remove the data point from its current location.
    let mut base_tree = tree.clone();
    if let Some(node_id) = current_node_id {
        if node_id == "-1" {
            base_tree.unassign_outlier(dp.idx);
        } else {
            let value = dp.value.clone();
            let node = base_tree
                .node_mut(node_id)
                .ok_or_else(|| format!("node {} not found", node_id))?;
            node.remove_data_point(dp.idx, &value)?;
            base_tree.update_path_to_root(node_id)?;
        }
    }

    // Propose assigning to every non-root node.
    let non_root_ids: Vec<String> = base_tree
        .non_root_node_ids()
        .map(|s| s.to_string())
        .collect();

    for target_node_id in &non_root_ids {
        let mut candidate = base_tree.clone();
        let node = candidate
            .node_mut(target_node_id)
            .ok_or_else(|| format!("target node {} not found", target_node_id))?;
        node.add_data_point(dp.idx, &dp.value)?;
        candidate.update_path_to_root(target_node_id)?;
        let score = joint.log_p_one_tree(&candidate, outlier_points)?;
        candidates.push((score, candidate));
    }

    // Optionally propose assigning to outliers.
    if outliers_active {
        let mut candidate = base_tree.clone();
        candidate.assign_outlier(dp.idx);
        let score = joint.log_p_one_tree(&candidate, outlier_points)?;
        candidates.push((score, candidate));
    }

    Ok(candidates)
}

// ── PruneRegraphSampler ───────────────────────────────────────────────────────

/// Prune a random subtree and re-attach it by Gibbs sampling.
///
/// Mirrors `PruneRegraphSampler.sample_tree` in `phyclone/mcmc/gibbs_mh.py`:
/// 1. Skip if ≤1 non-root node exists.
/// 2. Pick a random non-root node as subtree root.
/// 3. Detach the subtree.
/// 4. Enumerate all possible re-attachment points (every remaining node + root).
/// 5. Score each `log(num_children_before + 1) + log_p_one`.
/// 6. Sample proportional to scores.
pub struct CompatPruneRegraphSampler {
    pub joint: CompatTreeJointDistribution,
}

impl CompatPruneRegraphSampler {
    pub fn sample_tree(
        &self,
        data_points: &[CompatDataPoint],
        tree: &CompatTree,
        rng: &mut impl Rng,
    ) -> Result<CompatTree, String> {
        let outlier_points = CompatTree::outlier_points(data_points);

        // Collect non-root node ids.
        let non_root_ids: Vec<String> = tree.non_root_node_ids().map(|s| s.to_string()).collect();

        if non_root_ids.len() <= 1 {
            return Ok(tree.clone());
        }

        // Pick a random subtree root.
        let subtree_root_idx = rng.random_range(0..non_root_ids.len());
        let subtree_root_id = non_root_ids[subtree_root_idx].clone();

        // Collect the subtree node ids (subtree_root + all descendants).
        let subtree_nodes: HashSet<String> = {
            let mut set = HashSet::new();
            set.insert(subtree_root_id.clone());
            for d in tree.descendants_of(&subtree_root_id)? {
                set.insert(d);
            }
            set
        };

        // Remaining attachment candidates: every non-root node NOT in the subtree, plus
        // None (= attach directly under root).
        let mut remaining: Vec<Option<String>> = tree
            .non_root_node_ids()
            .filter(|id| !subtree_nodes.contains(*id))
            .map(|id| Some(id.to_string()))
            .collect();
        remaining.push(None); // attach under root

        if remaining.is_empty() {
            return Ok(tree.clone());
        }

        // Build a pruned tree (without the subtree).
        let mut pruned = tree.clone();
        pruned.remove_subtree(&subtree_root_id)?;

        // Build candidate trees for each attachment point.
        let mut candidates: Vec<(f64, CompatTree)> = Vec::new();
        for parent_opt in &remaining {
            let root_id = pruned.root_node_id.clone();
            let parent_id: &str = parent_opt.as_deref().unwrap_or(&root_id);
            let num_children_before = pruned.children_of(parent_id).unwrap_or_default().len();

            let mut candidate = pruned.clone();
            candidate.add_subtree(parent_id, &subtree_root_id)?;
            // Recompute the full tree to avoid stale values on the old-parent branch
            // after prune/regraph topology edits.
            candidate.update_all_nodes_postorder()?;

            let score = (num_children_before as f64 + 1.0).ln()
                + candidate.joint_log_p_one(&self.joint, &outlier_points)?;
            candidates.push((score, candidate));
        }

        let log_scores: Vec<f64> = candidates.iter().map(|(s, _)| *s).collect();
        let chosen = sample_discrete_from_log_weights(&log_scores, rng)?;
        Ok(candidates.into_iter().nth(chosen).unwrap().1)
    }
}

// ── ConcentrationSampler ──────────────────────────────────────────────────────

/// Gibbs update for the FS-CRP concentration parameter α, with a Gamma(a, b) prior.
///
/// Mirrors `GammaPriorConcentrationSampler.sample` in `phyclone/mcmc/concentration.py`.
pub struct CompatConcentrationSampler {
    pub a: f64,
    pub b: f64,
}

impl Default for CompatConcentrationSampler {
    fn default() -> Self {
        Self { a: 0.01, b: 0.01 }
    }
}

impl CompatConcentrationSampler {
    /// Sample a new concentration value.
    ///
    /// - `old_value`: current α  
    /// - `num_clusters`: number of non-outlier, non-empty nodes  
    /// - `num_data_points`: total non-outlier data points assigned  
    pub fn sample(
        &self,
        old_value: f64,
        num_clusters: usize,
        num_data_points: usize,
        rng: &mut impl Rng,
    ) -> f64 {
        if num_clusters == 0 {
            // Sample from prior Gamma(a, 1/b).
            return sample_gamma(self.a, 1.0 / self.b, rng).max(1e-10);
        }

        let a = self.a;
        let b = self.b;
        let k = num_clusters as f64;
        let n = num_data_points as f64;

        // eta ~ Beta(old_value + 1, n)
        let eta = sample_beta(old_value + 1.0, n, rng);

        let shape = a + k - 1.0;
        let rate = b - eta.ln();

        let x = shape / (n * rate);
        let pi = x / (1.0 + x);

        // Bernoulli(pi) adjustment to shape.
        let shape_adj = shape + sample_bernoulli(pi, rng) as f64;

        sample_gamma(shape_adj, 1.0 / rate, rng).max(1e-10)
    }
}

// ── RNG helpers (matching scipy.stats parameterisation) ───────────────────────

/// Sample from Gamma(shape, scale) using the Marsaglia–Tsang method.
/// `scale = 1 / rate`.
fn sample_gamma(shape: f64, scale: f64, rng: &mut impl Rng) -> f64 {
    if shape <= 0.0 || scale <= 0.0 {
        return 1e-10;
    }
    // Marsaglia-Tsang requires shape >= 1. For shape < 1, sample from
    // Gamma(shape + 1, scale) and transform with U^(1/shape).
    if shape < 1.0 {
        let u: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
        return sample_gamma(shape + 1.0, scale, rng) * u.powf(1.0 / shape);
    }
    // Marsaglia & Tsang (2000) fast Gamma sampler.
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        // Standard normal via Box-Muller.
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let v = (1.0 + c * z).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.random();
        if u < 1.0 - 0.0331 * (z * z) * (z * z) {
            return d * v * scale;
        }
        if u.ln() < 0.5 * z * z + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

/// Sample from Beta(a, b) using the ratio of two Gamma samples.
fn sample_beta(a: f64, b: f64, rng: &mut impl Rng) -> f64 {
    let x = sample_gamma(a, 1.0, rng);
    let y = sample_gamma(b, 1.0, rng);
    let sum = x + y;
    if sum == 0.0 {
        0.5
    } else {
        (x / sum).clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON)
    }
}

/// Sample from Bernoulli(p): returns 1 with probability p, 0 otherwise.
fn sample_bernoulli(p: f64, rng: &mut impl Rng) -> u32 {
    let u: f64 = rng.random();
    if u < p {
        1
    } else {
        0
    }
}

// ── CompatTree extension methods needed by PruneRegraphSampler ───────────────

impl CompatTree {
    /// Compute `log_p_one` for this tree using the provided joint distribution and outlier points.
    pub fn joint_log_p_one(
        &self,
        joint: &CompatTreeJointDistribution,
        outlier_points: &[CompatOutlierPoint],
    ) -> Result<f64, String> {
        joint.log_p_one_tree(self, outlier_points)
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phyclone::compat::distributions::CompatFscrpDistribution;
    use rand::SeedableRng;

    fn make_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(0)
    }

    fn make_joint() -> CompatTreeJointDistribution {
        CompatTreeJointDistribution {
            prior: CompatFscrpDistribution {
                alpha: 1.0,
                c_const: 1000.0,
            },
            outlier_modelling_active: false,
        }
    }

    fn make_dp(idx: usize) -> CompatDataPoint {
        use crate::phyclone::compat::data::CompatDataPointName;
        CompatDataPoint {
            idx,
            name: CompatDataPointName::Int(idx as i64),
            mutation_ids: vec![format!("m{}", idx)],
            sample_ids: vec!["s0".to_string()],
            value: vec![vec![-1.0, -1.5, -2.0]],
            raw_outlier_prob: f64::MIN_POSITIVE,
            outlier_prob: f64::MIN_POSITIVE.ln(),
            outlier_prob_not: 0.0,
            outlier_marginal_prob: -5.0,
            size: 1,
        }
    }

    #[test]
    fn concentration_sampler_returns_positive_value() {
        let sampler = CompatConcentrationSampler::default();
        let mut rng = make_rng();
        let v = sampler.sample(1.0, 3, 10, &mut rng);
        assert!(v > 0.0, "concentration must be positive, got {}", v);
    }

    #[test]
    fn data_point_sampler_preserves_all_data_points() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0 / 3.0_f64.ln(), 8);
        tree.add_child_node("root", "n0").unwrap();
        tree.add_child_node("root", "n1").unwrap();

        let dp0 = make_dp(0);
        let dp1 = make_dp(1);
        let dp2 = make_dp(2);

        let dps = vec![dp0.clone(), dp1.clone(), dp2.clone()];

        tree.node_mut("n0")
            .unwrap()
            .add_data_point(0, &dps[0].value)
            .unwrap();
        tree.node_mut("n0")
            .unwrap()
            .add_data_point(1, &dps[1].value)
            .unwrap();
        tree.node_mut("n1")
            .unwrap()
            .add_data_point(2, &dps[2].value)
            .unwrap();
        tree.update_path_to_root("n0").unwrap();
        tree.update_path_to_root("n1").unwrap();

        let sampler = CompatDataPointSampler {
            joint: make_joint(),
            outliers: false,
        };
        let mut rng = make_rng();
        let new_tree = sampler.sample_tree(&dps, &tree, &mut rng).unwrap();

        // All data points must still be assigned (in some node).
        for dp_id in [0, 1, 2] {
            assert!(
                new_tree.node_id_for_data_point(dp_id).is_some(),
                "dp {} missing from tree after DataPointSampler",
                dp_id
            );
        }
    }

    #[test]
    fn data_point_sampler_does_not_move_singleton_outlier() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0 / 3.0_f64.ln(), 8);
        tree.add_child_node("root", "n0").unwrap();

        let dp0 = make_dp(0);
        let dp1 = make_dp(1);
        let dps = vec![dp0.clone(), dp1.clone()];

        // n0 is singleton and outlier bucket is also singleton.
        tree.node_mut("n0")
            .unwrap()
            .add_data_point(1, &dps[1].value)
            .unwrap();
        tree.update_path_to_root("n0").unwrap();
        tree.assign_outlier(0);

        let sampler = CompatDataPointSampler {
            joint: make_joint(),
            outliers: true,
        };
        let mut rng = make_rng();
        let new_tree = sampler.sample_tree(&dps, &tree, &mut rng).unwrap();

        // With PhyClone-parity singleton guards, neither dp can move.
        assert_eq!(new_tree.node_id_for_data_point(0), Some("-1"));
        assert_eq!(new_tree.node_id_for_data_point(1), Some("n0"));
        assert_eq!(new_tree.assigned_outlier_ids().len(), 1);
    }
}
