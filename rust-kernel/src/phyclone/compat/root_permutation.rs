//! Rust port of PhyClone's `RootPermutationDistribution` (phyclone/smc/utils.py).
//!
//! `RootPermutationDistribution::sample` produces a random ordering of all data points
//! consistent with the tree topology, used as the data order for each SMC sweep.
//!
//! The algorithm:
//! 1. For each root (direct child of the virtual root node), recursively build a
//!    tree-consistent permutation of its subtree data points.
//! 2. "Bridge-shuffle" all root-subtree lists together (random interleaving).
//! 3. Bridge-shuffle the result with the outlier list.
//!
//! For each subtree rooted at `node`:
//! 1. Recurse into children and bridge-shuffle their permutation lists.
//! 2. Append the node's own data points in a random order.

#![allow(dead_code)]

use super::tree_ids::DataPointId;
use super::tree_model::CompatTree;
use rand::{Rng, RngExt};

/// Mirrors `RootPermutationDistribution` from PhyClone's `smc/utils.py`.
pub struct RootPermutationDistribution;

impl RootPermutationDistribution {
    /// Compute the log-count of valid data orderings for the given tree.
    ///
    /// Mirrors `RootPermutationDistribution.log_count(tree)` from PhyClone's `smc/utils.py`.
    ///
    /// The count is the number of distinct orderings consistent with the tree topology:
    /// - For each node: multinomial interleaving of children subtrees × factorial of node's own data
    /// - At root level: multinomial interleaving of root subtrees × binomial(total, num_outliers)
    pub fn log_count(tree: &CompatTree) -> f64 {
        let root_id = &tree.root_node_id;
        let roots: Vec<String> = tree
            .children_of(root_id)
            .unwrap_or_default()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let mut count = 0.0_f64;
        let mut subtree_sizes: Vec<usize> = Vec::new();

        for root_node in &roots {
            let (sub_count, sub_size) = log_count_subtree(tree, root_node);
            count += sub_count;
            subtree_sizes.push(sub_size);
        }

        // Bridge shuffle root nodes: log_multinomial_coefficient(subtree_sizes)
        count += log_multinomial_coefficient(&subtree_sizes);

        // Bridge shuffle outliers: log_binomial_coefficient(num_data_points, num_outliers)
        let num_data_points: usize =
            subtree_sizes.iter().sum::<usize>() + tree.assigned_outliers.len();
        let num_outliers = tree.assigned_outliers.len();
        count += log_binomial_coefficient(num_data_points, num_outliers);

        count
    }

    /// Compute the log-pdf of the root permutation distribution for the given tree.
    ///
    /// Mirrors `RootPermutationDistribution.log_pdf(tree)` from PhyClone's `smc/utils.py`:
    /// ```python
    /// return -RootPermutationDistribution.log_count(tree)
    /// ```
    pub fn log_pdf(tree: &CompatTree) -> f64 {
        -Self::log_count(tree)
    }

    /// Sample a tree-consistent permutation of all data point indices.
    ///
    /// Returns a `Vec<DataPointId>` whose length equals
    /// (number of non-outlier data points) + (number of outlier data points).
    pub fn sample(tree: &CompatTree, rng: &mut impl Rng) -> Vec<DataPointId> {
        // Collect direct children of the root (= "roots" in PhyClone terminology).
        let root_id = &tree.root_node_id;
        let roots: Vec<String> = tree
            .children_of(root_id)
            .unwrap_or_default()
            .iter()
            .map(|s| s.to_string())
            .collect();

        // Build a permutation for each root subtree.
        let mut root_sigmas: Vec<Vec<DataPointId>> = roots
            .iter()
            .map(|root_node| sample_subtree(tree, root_node, rng))
            .collect();

        // Bridge-shuffle all root-subtree permutations.
        let mut sigma = interleave_lists(&mut root_sigmas, rng);

        // Bridge-shuffle the result with the outlier list.
        let mut outliers: Vec<DataPointId> = tree.assigned_outliers.iter().copied().collect();
        // Shuffle the outlier list itself (mirrors `rng.shuffle(outliers)` in Python).
        fisher_yates_shuffle(&mut outliers, rng);
        let mut combined = vec![sigma, outliers];
        sigma = interleave_lists(&mut combined, rng);

        sigma
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// Recursively build a tree-consistent permutation for the subtree rooted at `node_id`.
fn sample_subtree(tree: &CompatTree, node_id: &str, rng: &mut impl Rng) -> Vec<DataPointId> {
    let children: Vec<String> = tree
        .children_of(node_id)
        .unwrap_or_default()
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Bridge-shuffle children sub-permutations.
    let mut child_sigmas: Vec<Vec<DataPointId>> = children
        .iter()
        .map(|child| sample_subtree(tree, child, rng))
        .collect();
    let mut sigma = interleave_lists(&mut child_sigmas, rng);

    // Append a shuffled copy of the node's own data points.
    let mut node_data: Vec<DataPointId> = tree
        .nodes
        .get(node_id)
        .map(|n| n.data_point_ids.clone())
        .unwrap_or_default();
    fisher_yates_shuffle(&mut node_data, rng);

    sigma.extend(node_data);
    sigma
}

/// "Bridge shuffle" (interleave_lists in PhyClone): given several lists, create a
/// uniformly-random interleaving by shuffling a list of sentinel indices and popping
/// from each source list in sentinel order.
///
/// Modifies `lists` in place (draining each sub-vec) and returns the merged result.
fn interleave_lists(lists: &mut [Vec<DataPointId>], rng: &mut impl Rng) -> Vec<DataPointId> {
    if lists.is_empty() {
        return Vec::new();
    }
    if lists.len() == 1 {
        return lists[0].drain(..).collect();
    }

    let total: usize = lists.iter().map(|l| l.len()).sum();
    if total == 0 {
        return Vec::new();
    }

    // Build sentinels: one entry per element indicating which sub-list it came from.
    let mut sentinels: Vec<usize> = lists
        .iter()
        .enumerate()
        .flat_map(|(i, l)| std::iter::repeat_n(i, l.len()))
        .collect();

    fisher_yates_shuffle(&mut sentinels, rng);

    // Consume elements from each sub-list in sentinel order.
    let mut pointers: Vec<usize> = vec![0; lists.len()];
    let mut result = Vec::with_capacity(total);
    for list_idx in sentinels {
        let ptr = &mut pointers[list_idx];
        result.push(lists[list_idx][*ptr]);
        *ptr += 1;
    }

    result
}

/// Fisher-Yates in-place shuffle (uniform random permutation).
/// Exported so `samplers.rs` can reuse it.
pub fn fisher_yates_shuffle_pub<T>(slice: &mut [T], rng: &mut impl Rng) {
    fisher_yates_shuffle(slice, rng);
}

/// Fisher-Yates in-place shuffle (uniform random permutation).
fn fisher_yates_shuffle<T>(slice: &mut [T], rng: &mut impl Rng) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        slice.swap(i, j);
    }
}

/// Recursively compute (log_count, subtree_data_len) for the subtree rooted at `node_id`.
///
/// Mirrors the `source is not None` branch of `RootPermutationDistribution.log_count`.
fn log_count_subtree(tree: &CompatTree, node_id: &str) -> (f64, usize) {
    let children: Vec<String> = tree
        .children_of(node_id)
        .unwrap_or_default()
        .iter()
        .map(|s| s.to_string())
        .collect();

    let mut count = 0.0_f64;
    let mut child_subtree_sizes: Vec<usize> = Vec::new();

    for child in &children {
        let (child_count, child_size) = log_count_subtree(tree, child);
        count += child_count;
        child_subtree_sizes.push(child_size);
    }

    // Bridge shuffle children: log_multinomial_coefficient(child_subtree_sizes)
    count += log_multinomial_coefficient(&child_subtree_sizes);

    // Permute the node's own data: log_factorial(node_data_len)
    let node_data_len = tree
        .nodes
        .get(node_id)
        .map(|n| n.data_point_ids.len())
        .unwrap_or(0);
    count += log_factorial(node_data_len);

    let subtree_size: usize = child_subtree_sizes.iter().sum::<usize>() + node_data_len;
    (count, subtree_size)
}

/// log(n!) using lgamma.
fn log_factorial(n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    (1..=n).map(|k| (k as f64).ln()).sum()
}

/// log(C(n, k)) = log(n!) - log(k!) - log((n-k)!)
fn log_binomial_coefficient(n: usize, k: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    log_factorial(n) - log_factorial(k) - log_factorial(n - k)
}

/// log(n! / (k1! * k2! * ... * km!)) where sum(ki) = n
fn log_multinomial_coefficient(sizes: &[usize]) -> f64 {
    if sizes.len() <= 1 {
        return 0.0;
    }
    let n: usize = sizes.iter().sum();
    log_factorial(n) - sizes.iter().map(|&k| log_factorial(k)).sum::<f64>()
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phyclone::compat::tree_model::CompatTree;
    use rand::SeedableRng;

    fn make_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

    fn add_dp(tree: &mut CompatTree, node: &str, dp_id: usize) {
        let value = vec![vec![0.0; 3]];
        tree.node_mut(node)
            .unwrap()
            .add_data_point(dp_id, &value)
            .unwrap();
    }

    /// Single node tree: sigma is just the node's data (shuffled).
    #[test]
    fn single_node_tree_returns_all_data() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "n0").unwrap();
        add_dp(&mut tree, "n0", 0);
        add_dp(&mut tree, "n0", 1);
        add_dp(&mut tree, "n0", 2);

        let mut rng = make_rng();
        let sigma = RootPermutationDistribution::sample(&tree, &mut rng);
        let mut sorted = sigma.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    /// Two root nodes: sigma contains all data from both subtrees.
    #[test]
    fn two_root_nodes_return_all_data() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "n0").unwrap();
        tree.add_child_node("root", "n1").unwrap();
        add_dp(&mut tree, "n0", 0);
        add_dp(&mut tree, "n0", 1);
        add_dp(&mut tree, "n1", 2);
        add_dp(&mut tree, "n1", 3);

        let mut rng = make_rng();
        let sigma = RootPermutationDistribution::sample(&tree, &mut rng);
        let mut sorted = sigma.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    /// Nested tree: root -> n0 -> n1.  All data must appear.
    #[test]
    fn nested_tree_returns_all_data() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "n0").unwrap();
        tree.add_child_node("n0", "n1").unwrap();
        add_dp(&mut tree, "n0", 0);
        add_dp(&mut tree, "n1", 1);
        add_dp(&mut tree, "n1", 2);

        let mut rng = make_rng();
        let sigma = RootPermutationDistribution::sample(&tree, &mut rng);
        let mut sorted = sigma.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    /// Outlier data points must be included in sigma.
    #[test]
    fn outlier_data_points_included_in_sigma() {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "n0").unwrap();
        add_dp(&mut tree, "n0", 0);
        tree.assign_outlier(1);
        tree.assign_outlier(2);

        let mut rng = make_rng();
        let sigma = RootPermutationDistribution::sample(&tree, &mut rng);
        let mut sorted = sigma.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2]);
    }

    /// Empty tree (no data): sigma is empty.
    #[test]
    fn empty_tree_returns_empty_sigma() {
        let tree = CompatTree::new("root", 1, 3, -1.0, 8);
        let mut rng = make_rng();
        let sigma = RootPermutationDistribution::sample(&tree, &mut rng);
        assert!(sigma.is_empty());
    }
}
