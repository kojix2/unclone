#![allow(dead_code)]

use super::tree_ids::DataPointId;
use super::tree_ids::NodeId;
use super::tree_model::CompatTree;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompatProposalKind {
    ExistingNode { node_id: NodeId },
    NewNode { children: Vec<NodeId> },
    Outlier,
}

#[derive(Clone, Debug)]
pub struct CompatProposalCandidate {
    pub kind: CompatProposalKind,
    pub tree: CompatTree,
}

#[derive(Clone, Debug)]
pub struct CompatProposalSet {
    pub candidates: Vec<CompatProposalCandidate>,
    pub log_q: Vec<f64>,
}

impl CompatProposalSet {
    pub fn len(&self) -> usize {
        self.candidates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub fn log_q_of(&self, idx: usize) -> Option<f64> {
        self.log_q.get(idx).copied()
    }
}

#[derive(Clone, Debug)]
pub struct CompatTreeShellNodeAdder {
    pub outlier_node_id: NodeId,
    pub new_node_prefix: String,
}

impl Default for CompatTreeShellNodeAdder {
    fn default() -> Self {
        Self {
            outlier_node_id: "-1".to_string(),
            new_node_prefix: "shell".to_string(),
        }
    }
}

impl CompatTreeShellNodeAdder {
    pub fn ensure_outlier_node(&self, tree: &mut CompatTree) -> Result<(), String> {
        if tree.node(&self.outlier_node_id).is_some() {
            return Ok(());
        }
        tree.add_child_node(&tree.root_node_id.clone(), self.outlier_node_id.clone())
    }

    pub fn root_children_without_outlier(&self, tree: &CompatTree) -> Result<Vec<NodeId>, String> {
        let children = tree.children_of(&tree.root_node_id)?;
        Ok(children
            .into_iter()
            .filter(|id| *id != self.outlier_node_id)
            .map(|id| id.to_string())
            .collect())
    }

    pub fn create_tree_with_datapoint_added_to_node(
        &self,
        tree: &CompatTree,
        node_id: &str,
        data_point_id: DataPointId,
        data_point_value: &[Vec<f64>],
    ) -> Result<CompatTree, String> {
        let mut new_tree = tree.clone();
        let node = new_tree
            .node_mut(node_id)
            .ok_or_else(|| "target node does not exist".to_string())?;
        node.add_data_point(data_point_id, data_point_value)?;
        new_tree.update_path_to_root(node_id)?;
        Ok(new_tree)
    }

    pub fn create_tree_with_datapoint_added_to_outliers(
        &self,
        tree: &CompatTree,
        data_point_id: DataPointId,
        _data_point_value: &[Vec<f64>],
    ) -> Result<CompatTree, String> {
        // Outliers are tracked in `assigned_outliers`, not as a graph node.
        // This matches the Python PhyClone design where outlier data points are stored in
        // `self._data[-1]` (outside the graph) and do NOT contribute to `tree_likelihood`.
        let mut new_tree = tree.clone();
        new_tree.assign_outlier(data_point_id);
        Ok(new_tree)
    }

    pub fn create_tree_with_new_node(
        &self,
        tree: &CompatTree,
        children: &[NodeId],
        data_point_id: DataPointId,
        data_point_value: &[Vec<f64>],
    ) -> Result<CompatTree, String> {
        let mut new_tree = tree.clone();
        let root_id = new_tree.root_node_id.clone();

        for child in children {
            let parent = new_tree.parent_of(child)?;
            if parent != Some(root_id.as_str()) {
                return Err("new-node children must be current roots".to_string());
            }
        }

        let new_node_id = self.allocate_new_node_id(&new_tree);
        new_tree.add_child_node(&root_id, new_node_id.clone())?;

        for child in children {
            new_tree.remove_subtree(child)?;
            new_tree.add_subtree(&new_node_id, child)?;
        }

        self.create_tree_with_datapoint_added_to_node(
            &new_tree,
            &new_node_id,
            data_point_id,
            data_point_value,
        )
    }

    fn allocate_new_node_id(&self, tree: &CompatTree) -> NodeId {
        let mut idx = tree.nodes.len();
        loop {
            let candidate = format!("{}-{}", self.new_node_prefix, idx);
            if tree.node(&candidate).is_none() {
                return candidate;
            }
            idx += 1;
        }
    }
}

/// Build a bootstrap proposal set for the given data point.
///
/// **Self-consistent proposal distribution**: the discrete sampling weights
/// stored in `CompatProposalSet::log_q` are computed by [`bootstrap_log_q`]
/// and are *identical* to the distribution actually used when a caller samples
/// from the set.  This means the importance weight `log w = log γ(x') - log q(x')`
/// is always well-defined and unbiased.
///
/// # Difference from upstream PhyClone
///
/// PhyClone's `BootstrapProposalDistribution` has an apparent implementation
/// bug: `sample()` draws uniformly over all candidates (ignoring the outlier
/// prior weight) while `log_p()` returns the non-uniform log-probability that
/// includes the outlier penalty.  When `outlier_modelling_active` is `true`
/// this mismatch makes the importance weights incorrect in the upstream code.
/// tyclone intentionally **does not** reproduce that bug; the sampling
/// distribution and `log_q` are always consistent here.
pub fn build_bootstrap_proposal_set(
    tree: &CompatTree,
    adder: &CompatTreeShellNodeAdder,
    data_point_id: DataPointId,
    data_point_value: &[Vec<f64>],
    outlier_modelling_active: bool,
    parent_particle_is_none: bool,
) -> Result<CompatProposalSet, String> {
    let roots = adder.root_children_without_outlier(tree)?;
    let mut kinds = Vec::new();

    if outlier_modelling_active {
        kinds.push(CompatProposalKind::Outlier);
    }

    for root in &roots {
        kinds.push(CompatProposalKind::ExistingNode {
            node_id: root.clone(),
        });
    }

    for children in enumerate_new_node_children(&roots) {
        kinds.push(CompatProposalKind::NewNode { children });
    }

    let mut candidates = Vec::with_capacity(kinds.len());
    let mut log_q = Vec::with_capacity(kinds.len());

    for kind in kinds {
        let proposal_tree = match &kind {
            CompatProposalKind::Outlier => adder.create_tree_with_datapoint_added_to_outliers(
                tree,
                data_point_id,
                data_point_value,
            )?,
            CompatProposalKind::ExistingNode { node_id } => adder
                .create_tree_with_datapoint_added_to_node(
                    tree,
                    node_id,
                    data_point_id,
                    data_point_value,
                )?,
            CompatProposalKind::NewNode { children } => {
                adder.create_tree_with_new_node(tree, children, data_point_id, data_point_value)?
            }
        };

        let q = bootstrap_log_q(
            &kind,
            roots.len(),
            outlier_modelling_active,
            parent_particle_is_none,
        )?;

        candidates.push(CompatProposalCandidate {
            kind,
            tree: proposal_tree,
        });
        log_q.push(q);
    }

    Ok(CompatProposalSet { candidates, log_q })
}

pub fn build_fully_adapted_proposal_set<F>(
    tree: &CompatTree,
    adder: &CompatTreeShellNodeAdder,
    data_point_id: DataPointId,
    data_point_value: &[Vec<f64>],
    outlier_modelling_active: bool,
    scorer: F,
) -> Result<CompatProposalSet, String>
where
    F: Fn(&CompatTree) -> f64,
{
    let roots = adder.root_children_without_outlier(tree)?;
    let kinds = enumerate_fully_adapted_candidates(&roots, outlier_modelling_active);

    let mut candidates = Vec::with_capacity(kinds.len());
    let mut log_scores = Vec::with_capacity(kinds.len());

    for kind in kinds {
        let proposal_tree = match &kind {
            CompatProposalKind::Outlier => adder.create_tree_with_datapoint_added_to_outliers(
                tree,
                data_point_id,
                data_point_value,
            )?,
            CompatProposalKind::ExistingNode { node_id } => adder
                .create_tree_with_datapoint_added_to_node(
                    tree,
                    node_id,
                    data_point_id,
                    data_point_value,
                )?,
            CompatProposalKind::NewNode { children } => {
                adder.create_tree_with_new_node(tree, children, data_point_id, data_point_value)?
            }
        };

        log_scores.push(scorer(&proposal_tree));
        candidates.push(CompatProposalCandidate {
            kind,
            tree: proposal_tree,
        });
    }

    let log_q = normalize_log_scores(&log_scores)?;
    Ok(CompatProposalSet { candidates, log_q })
}

pub fn build_semi_adapted_proposal_set<F>(
    tree: &CompatTree,
    adder: &CompatTreeShellNodeAdder,
    data_point_id: DataPointId,
    data_point_value: &[Vec<f64>],
    outlier_modelling_active: bool,
    scorer: F,
) -> Result<CompatProposalSet, String>
where
    F: Fn(&CompatTree) -> f64,
{
    let roots = adder.root_children_without_outlier(tree)?;

    // PhyClone compatibility for parent empty tree:
    // candidates are NewNode{[]} and (optionally) Outlier, and q is obtained
    // by normalizing target scores over those candidates.
    if roots.is_empty() {
        let children: Vec<NodeId> = Vec::new();
        let new_tree =
            adder.create_tree_with_new_node(tree, &children, data_point_id, data_point_value)?;

        let mut candidates = vec![CompatProposalCandidate {
            kind: CompatProposalKind::NewNode { children },
            tree: new_tree,
        }];
        let mut log_scores = vec![scorer(&candidates[0].tree)];

        if outlier_modelling_active {
            let outlier_tree = adder.create_tree_with_datapoint_added_to_outliers(
                tree,
                data_point_id,
                data_point_value,
            )?;
            candidates.push(CompatProposalCandidate {
                kind: CompatProposalKind::Outlier,
                tree: outlier_tree,
            });
            log_scores.push(scorer(&candidates[1].tree));
        }

        let log_q = normalize_log_scores(&log_scores)?;
        return Ok(CompatProposalSet { candidates, log_q });
    }

    let mut existing_or_outlier: Vec<CompatProposalCandidate> = Vec::new();
    let mut existing_scores = Vec::new();

    if outlier_modelling_active {
        let outlier_tree = adder.create_tree_with_datapoint_added_to_outliers(
            tree,
            data_point_id,
            data_point_value,
        )?;
        existing_scores.push(scorer(&outlier_tree));
        existing_or_outlier.push(CompatProposalCandidate {
            kind: CompatProposalKind::Outlier,
            tree: outlier_tree,
        });
    }

    for root in &roots {
        let t = adder.create_tree_with_datapoint_added_to_node(
            tree,
            root,
            data_point_id,
            data_point_value,
        )?;
        existing_scores.push(scorer(&t));
        existing_or_outlier.push(CompatProposalCandidate {
            kind: CompatProposalKind::ExistingNode {
                node_id: root.clone(),
            },
            tree: t,
        });
    }

    let existing_log_q = normalize_log_scores(&existing_scores)?;

    let mut candidates = Vec::new();
    let mut log_q = Vec::new();

    for (cand, q_existing) in existing_or_outlier.into_iter().zip(existing_log_q) {
        candidates.push(cand);
        log_q.push(0.5_f64.ln() + q_existing);
    }

    for children in enumerate_new_node_children(&roots) {
        let tree_new =
            adder.create_tree_with_new_node(tree, &children, data_point_id, data_point_value)?;
        let q = semi_adapted_new_node_log_q(roots.len(), children.len())?;
        candidates.push(CompatProposalCandidate {
            kind: CompatProposalKind::NewNode { children },
            tree: tree_new,
        });
        log_q.push(q);
    }

    Ok(CompatProposalSet { candidates, log_q })
}

pub fn enumerate_new_node_children(roots: &[NodeId]) -> Vec<Vec<NodeId>> {
    let n = roots.len();
    let mut out = Vec::new();

    for r in 0..=n {
        let mut curr = Vec::with_capacity(r);
        enumerate_combinations_recursive(roots, 0, r, &mut curr, &mut out);
    }

    out
}

pub fn enumerate_fully_adapted_candidates(
    roots: &[NodeId],
    outlier_modelling_active: bool,
) -> Vec<CompatProposalKind> {
    let mut candidates = Vec::new();

    // Python ordering parity: new-node trees first.
    for children in enumerate_new_node_children(roots) {
        candidates.push(CompatProposalKind::NewNode { children });
    }

    // Then outlier if active.
    if outlier_modelling_active {
        candidates.push(CompatProposalKind::Outlier);
    }

    // Then existing-node trees in root order.
    for root in roots {
        candidates.push(CompatProposalKind::ExistingNode {
            node_id: root.clone(),
        });
    }

    candidates
}

/// Compute the log proposal probability `log q(x')` for a bootstrap proposal candidate.
///
/// The returned value is the *exact* log-probability under the discrete distribution
/// that [`build_bootstrap_proposal_set`] uses when sampling a candidate.  Callers
/// may therefore use it directly as the denominator in an importance weight without
/// any correction.
///
/// When `outlier_modelling_active` is `true`, the outlier candidate receives weight
/// `0.1` and the remaining structural candidates share weight `0.9`; this split is
/// reflected both here and in the actual sampler.  This is intentionally **different**
/// from PhyClone's upstream behaviour, where `sample()` ignores the outlier weight
/// while `log_p()` accounts for it, producing inconsistent importance weights.
pub fn bootstrap_log_q(
    proposal_kind: &CompatProposalKind,
    num_roots: usize,
    outlier_modelling_active: bool,
    parent_particle_is_none: bool,
) -> Result<f64, String> {
    let outlier_prob: f64 = if outlier_modelling_active {
        0.1_f64
    } else {
        0.0_f64
    };

    if parent_particle_is_none {
        return match proposal_kind {
            CompatProposalKind::Outlier => {
                if outlier_modelling_active {
                    Ok(outlier_prob.ln())
                } else {
                    Err("outlier proposal is disabled".to_string())
                }
            }
            CompatProposalKind::ExistingNode { .. } | CompatProposalKind::NewNode { .. } => {
                Ok((1.0_f64 - outlier_prob).ln())
            }
        };
    }

    match proposal_kind {
        CompatProposalKind::Outlier => {
            if outlier_modelling_active {
                Ok(outlier_prob.ln())
            } else {
                Err("outlier proposal is disabled".to_string())
            }
        }
        CompatProposalKind::ExistingNode { .. } => {
            if num_roots == 0 {
                return Err("existing-node proposal requires num_roots > 0".to_string());
            }
            Ok(((1.0_f64 - outlier_prob) / 2.0_f64).ln() - (num_roots as f64).ln())
        }
        CompatProposalKind::NewNode { children } => {
            if children.len() > num_roots {
                return Err("new-node children count must be <= num_roots".to_string());
            }

            let comb = log_binomial_coefficient(num_roots, children.len())?;
            Ok(((1.0_f64 - outlier_prob) / 2.0_f64).ln() - ((num_roots + 1) as f64).ln() - comb)
        }
    }
}

pub fn semi_adapted_new_node_log_q(num_roots: usize, num_children: usize) -> Result<f64, String> {
    if num_children > num_roots {
        return Err("num_children must be <= num_roots".to_string());
    }

    let log_half = 0.5_f64.ln();
    let comb = log_binomial_coefficient(num_roots, num_children)?;
    Ok(log_half - ((num_roots + 1) as f64).ln() - comb)
}

pub fn normalize_log_scores(log_scores: &[f64]) -> Result<Vec<f64>, String> {
    if log_scores.is_empty() {
        return Err("log_scores must not be empty".to_string());
    }

    let max = log_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() && max.is_sign_negative() {
        return Err("all log scores are -inf".to_string());
    }

    let sum_exp: f64 = log_scores.iter().map(|v| (*v - max).exp()).sum();
    let lse = max + sum_exp.ln();
    Ok(log_scores.iter().map(|v| *v - lse).collect())
}

fn enumerate_combinations_recursive(
    roots: &[NodeId],
    start: usize,
    k: usize,
    current: &mut Vec<NodeId>,
    out: &mut Vec<Vec<NodeId>>,
) {
    if current.len() == k {
        out.push(current.clone());
        return;
    }

    let need = k - current.len();
    if need == 0 {
        out.push(current.clone());
        return;
    }

    for idx in start..=roots.len().saturating_sub(need) {
        current.push(roots[idx].clone());
        enumerate_combinations_recursive(roots, idx + 1, k, current, out);
        current.pop();
    }
}

fn log_binomial_coefficient(n: usize, k: usize) -> Result<f64, String> {
    if k > n {
        return Err("k must be <= n".to_string());
    }

    Ok(log_factorial(n) - log_factorial(k) - log_factorial(n - k))
}

fn log_factorial(n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    (2..=n).fold(0.0, |acc, x| acc + (x as f64).ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phyclone::compat::tree_model::CompatTree;

    fn roots() -> Vec<NodeId> {
        vec!["r0".to_string(), "r1".to_string(), "r2".to_string()]
    }

    fn approx_eq(lhs: f64, rhs: f64) {
        let delta = (lhs - rhs).abs();
        assert!(delta <= 1e-10, "lhs={} rhs={} delta={}", lhs, rhs, delta);
    }

    #[test]
    fn enumerate_new_node_children_matches_python_combination_order() {
        let out = enumerate_new_node_children(&roots());

        let expected = vec![
            vec![],
            vec!["r0".to_string()],
            vec!["r1".to_string()],
            vec!["r2".to_string()],
            vec!["r0".to_string(), "r1".to_string()],
            vec!["r0".to_string(), "r2".to_string()],
            vec!["r1".to_string(), "r2".to_string()],
            vec!["r0".to_string(), "r1".to_string(), "r2".to_string()],
        ];

        assert_eq!(out, expected);
    }

    #[test]
    fn enumerate_fully_adapted_candidates_has_expected_phase_order() {
        let out = enumerate_fully_adapted_candidates(&roots(), true);

        assert_eq!(out[0], CompatProposalKind::NewNode { children: vec![] });
        assert!(matches!(out[8], CompatProposalKind::Outlier));
        assert_eq!(
            out[9],
            CompatProposalKind::ExistingNode {
                node_id: "r0".to_string()
            }
        );
        assert_eq!(
            out[10],
            CompatProposalKind::ExistingNode {
                node_id: "r1".to_string()
            }
        );
        assert_eq!(
            out[11],
            CompatProposalKind::ExistingNode {
                node_id: "r2".to_string()
            }
        );
    }

    #[test]
    fn bootstrap_log_q_matches_python_formula_for_existing_new_outlier() {
        let existing = CompatProposalKind::ExistingNode {
            node_id: "r0".to_string(),
        };
        let new_node = CompatProposalKind::NewNode {
            children: vec!["r0".to_string(), "r1".to_string()],
        };
        let outlier = CompatProposalKind::Outlier;

        let log_q_existing =
            bootstrap_log_q(&existing, 3, true, false).expect("existing log_q should compute");
        let log_q_new =
            bootstrap_log_q(&new_node, 3, true, false).expect("new-node log_q should compute");
        let log_q_out =
            bootstrap_log_q(&outlier, 3, true, false).expect("outlier log_q should compute");

        let expected_existing = (0.45_f64).ln() - (3_f64).ln();
        let expected_new = (0.45_f64).ln() - (4_f64).ln() - (3_f64).ln();
        let expected_out = (0.1_f64).ln();

        approx_eq(log_q_existing, expected_existing);
        approx_eq(log_q_new, expected_new);
        approx_eq(log_q_out, expected_out);
    }

    #[test]
    fn bootstrap_first_particle_matches_python_behavior() {
        let new_node = CompatProposalKind::NewNode { children: vec![] };
        let outlier = CompatProposalKind::Outlier;

        let log_q_new =
            bootstrap_log_q(&new_node, 0, true, true).expect("new-node log_q should compute");
        let log_q_out =
            bootstrap_log_q(&outlier, 0, true, true).expect("outlier log_q should compute");

        approx_eq(log_q_new, (0.9_f64).ln());
        approx_eq(log_q_out, (0.1_f64).ln());
    }

    #[test]
    fn semi_adapted_new_node_log_q_matches_python_formula() {
        let actual = semi_adapted_new_node_log_q(4, 2).expect("semi log_q should compute");
        let expected = (0.5_f64).ln() - (5_f64).ln() - (6_f64).ln();
        approx_eq(actual, expected);
    }

    #[test]
    fn normalize_log_scores_returns_log_probs_that_sum_to_one() {
        let normalized =
            normalize_log_scores(&[-2.0, -0.5, -1.0]).expect("normalization should work");
        let prob_sum: f64 = normalized.iter().map(|v| v.exp()).sum();
        approx_eq(prob_sum, 1.0);
    }

    fn make_tree() -> CompatTree {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "r0")
            .expect("add r0 should succeed");
        tree.add_child_node("root", "r1")
            .expect("add r1 should succeed");
        tree
    }

    #[test]
    fn tree_shell_node_adder_can_add_new_node_and_reparent_children() {
        let tree = make_tree();
        let adder = CompatTreeShellNodeAdder {
            outlier_node_id: "-1".to_string(),
            new_node_prefix: "new".to_string(),
        };

        let new_tree = adder
            .create_tree_with_new_node(&tree, &["r0".to_string()], 42, &[vec![0.0; 3]])
            .expect("create_tree_with_new_node should succeed");

        let roots = adder
            .root_children_without_outlier(&new_tree)
            .expect("root children should load");
        assert!(roots.contains(&"r1".to_string()));
        assert_eq!(
            new_tree.parent_of("r0").expect("r0 must exist"),
            Some("new-3")
        );
        assert_eq!(new_tree.multiplicity("new-3").expect("new-3 must exist"), 1);
    }

    #[test]
    fn build_bootstrap_proposal_set_produces_finite_log_q() {
        let tree = make_tree();
        let adder = CompatTreeShellNodeAdder {
            outlier_node_id: "-1".to_string(),
            new_node_prefix: "new".to_string(),
        };

        let set = build_bootstrap_proposal_set(&tree, &adder, 10, &[vec![0.0; 3]], true, false)
            .expect("bootstrap proposal set should build");
        assert!(!set.is_empty());
        assert_eq!(set.candidates.len(), set.log_q.len());
        assert!(set.log_q.iter().all(|q| q.is_finite()));
    }

    #[test]
    fn build_fully_adapted_and_semi_sets_produce_probabilities() {
        let tree = make_tree();
        let adder = CompatTreeShellNodeAdder {
            outlier_node_id: "-1".to_string(),
            new_node_prefix: "new".to_string(),
        };

        let scorer = |t: &CompatTree| -> f64 { -(t.nodes.len() as f64) };

        let full =
            build_fully_adapted_proposal_set(&tree, &adder, 11, &[vec![0.0; 3]], true, scorer)
                .expect("fully-adapted set should build");
        assert!(!full.is_empty());
        let full_prob_sum: f64 = full.log_q.iter().map(|v| v.exp()).sum();
        approx_eq(full_prob_sum, 1.0);

        let semi =
            build_semi_adapted_proposal_set(&tree, &adder, 12, &[vec![0.0; 3]], true, scorer)
                .expect("semi-adapted set should build");
        assert!(!semi.is_empty());
        assert!(semi.log_q.iter().all(|q| q.is_finite()));
    }

    // ──────────────────────────────────────────────────────────────────────────────
    // Oracle tests: compare tyclone log_q with values derived directly from
    // PhyClone's proposal formulas.  Each test fixes a small tree, uses a constant
    // scorer (returns 0.0), and checks every candidate's log_q to 1e-12 precision.
    // ──────────────────────────────────────────────────────────────────────────────

    fn approx_eq_strict(lhs: f64, rhs: f64, label: &str) {
        let delta = (lhs - rhs).abs();
        assert!(
            delta <= 1e-12,
            "{}: lhs={:.15e} rhs={:.15e} delta={:.3e}",
            label,
            lhs,
            rhs,
            delta
        );
    }

    /// Build a tree with `n` root children named "r0".."rN-1".
    fn make_n_root_tree(n: usize) -> CompatTree {
        let mut tree = CompatTree::new("root", 1, 3, 0.0, 100);
        for i in 0..n {
            tree.add_child_node("root", format!("r{}", i))
                .unwrap_or_else(|_| panic!("add r{} failed", i));
        }
        tree
    }

    /// Constant scorer – always returns 0.0.  Gives uniform scores over all
    /// existing-node / outlier candidates so normalized probabilities equal -log(N).
    fn const_scorer(_: &CompatTree) -> f64 {
        0.0
    }

    // ── Semi-adapted: empty tree, outlier inactive ────────────────────────────────
    // PhyClone: parent_is_empty_tree=True, trees=[NewNode{[]}], log_p[NewNode]=0.0
    #[test]
    fn oracle_semi_adapted_empty_tree_outlier_inactive() {
        let tree = make_n_root_tree(0);
        let adder = CompatTreeShellNodeAdder::default();

        let set =
            build_semi_adapted_proposal_set(&tree, &adder, 0, &[vec![0.0; 3]], false, const_scorer)
                .expect("should build");

        assert_eq!(set.candidates.len(), 1, "exactly one candidate");
        assert!(
            matches!(&set.candidates[0].kind, CompatProposalKind::NewNode { children } if children.is_empty()),
            "must be NewNode{{[]}}"
        );
        approx_eq_strict(set.log_q[0], 0.0, "log_q[NewNode{{[]}}]");
    }

    // ── Semi-adapted: empty tree, outlier active ──────────────────────────────────
    // PhyClone: parent_is_empty_tree=True, trees=[NewNode{[]}, Outlier],
    //   both scored 0.0 → normalized = log(0.5) each.
    // Tyclone path: existing_or_outlier=[Outlier] → normalized=[0.0],
    //   Outlier→log_half+0.0=log(0.5); NewNode{[]} via semi_new_log_q(0,0)=log(0.5).
    #[test]
    fn oracle_semi_adapted_empty_tree_outlier_active() {
        let tree = make_n_root_tree(0);
        let adder = CompatTreeShellNodeAdder::default();

        let set =
            build_semi_adapted_proposal_set(&tree, &adder, 0, &[vec![0.0; 3]], true, const_scorer)
                .expect("should build");

        assert_eq!(set.candidates.len(), 2);
        let log_half = 0.5_f64.ln();

        // both candidates must have log_q = log(0.5)
        for (i, q) in set.log_q.iter().enumerate() {
            approx_eq_strict(*q, log_half, &format!("log_q[{}]", i));
        }
        // sum = 1.0
        let sum: f64 = set.log_q.iter().map(|v| v.exp()).sum();
        approx_eq(sum, 1.0);
    }

    #[test]
    fn oracle_semi_adapted_empty_tree_outlier_active_uses_target_score() {
        let tree = make_n_root_tree(0);
        let adder = CompatTreeShellNodeAdder::default();

        // Bias by node count so NewNode and Outlier receive different scores.
        let scorer = |t: &CompatTree| -> f64 { -(t.nodes.len() as f64) };

        let set = build_semi_adapted_proposal_set(&tree, &adder, 0, &[vec![0.0; 3]], true, scorer)
            .expect("should build");
        assert_eq!(set.candidates.len(), 2);

        let children: Vec<NodeId> = Vec::new();
        let new_tree = adder
            .create_tree_with_new_node(&tree, &children, 0, &[vec![0.0; 3]])
            .expect("new-node tree should build");
        let outlier_tree = adder
            .create_tree_with_datapoint_added_to_outliers(&tree, 0, &[vec![0.0; 3]])
            .expect("outlier tree should build");

        let expected = normalize_log_scores(&[scorer(&new_tree), scorer(&outlier_tree)])
            .expect("normalization should work");

        let mut q_new = None;
        let mut q_outlier = None;
        for (cand, q) in set.candidates.iter().zip(set.log_q.iter()) {
            match cand.kind {
                CompatProposalKind::NewNode { .. } => q_new = Some(*q),
                CompatProposalKind::Outlier => q_outlier = Some(*q),
                CompatProposalKind::ExistingNode { .. } => {}
            }
        }

        let q_new = q_new.expect("new-node candidate must exist");
        let q_outlier = q_outlier.expect("outlier candidate must exist");

        approx_eq_strict(q_new, expected[0], "empty-tree/new-node");
        approx_eq_strict(q_outlier, expected[1], "empty-tree/outlier");
        assert!(
            (q_new - q_outlier).abs() > 1e-12,
            "expected non-uniform q for non-uniform scorer"
        );
    }

    // ── Semi-adapted: 2 roots, outlier inactive, constant scorer ─────────────────
    // PhyClone formulas:
    //   existing  : log(0.5) + normalized_score = log(0.5) - log(2)
    //   NewNode{k}: log(0.5) - log(num_roots+1) - log(C(num_roots,k))
    //             = log(0.5) - log(3) - log(C(2,k))
    #[test]
    fn oracle_semi_adapted_two_roots_outlier_inactive() {
        let tree = make_n_root_tree(2); // roots: r0, r1
        let adder = CompatTreeShellNodeAdder::default();

        let set = build_semi_adapted_proposal_set(
            &tree,
            &adder,
            99,
            &[vec![0.0; 3]],
            false,
            const_scorer,
        )
        .expect("should build");

        // 2 existing + 4 new-node combos (k=0,1,1,2) = 6 total
        assert_eq!(set.candidates.len(), 6);

        let lh = 0.5_f64.ln(); // log(0.5)

        // indices 0,1 = ExistingNode{r0}, ExistingNode{r1}
        // normalized score over 2 existing = -log(2) each
        approx_eq_strict(set.log_q[0], lh - 2_f64.ln(), "existing r0");
        approx_eq_strict(set.log_q[1], lh - 2_f64.ln(), "existing r1");

        // indices 2..5 = NewNode{[]}, NewNode{[r0]}, NewNode{[r1]}, NewNode{[r0,r1]}
        // semi_new_log_q(2,0) = lh - log(3) - log(C(2,0)) = lh - log(3)
        approx_eq_strict(set.log_q[2], lh - 3_f64.ln(), "NewNode{{[]}}");
        // semi_new_log_q(2,1) = lh - log(3) - log(2)
        approx_eq_strict(
            set.log_q[3],
            lh - 3_f64.ln() - 2_f64.ln(),
            "NewNode{{[r0]}}",
        );
        approx_eq_strict(
            set.log_q[4],
            lh - 3_f64.ln() - 2_f64.ln(),
            "NewNode{{[r1]}}",
        );
        // semi_new_log_q(2,2) = lh - log(3) - log(C(2,2)) = lh - log(3)
        approx_eq_strict(set.log_q[5], lh - 3_f64.ln(), "NewNode{{[r0,r1]}}");

        // probability sum must be 1.0
        let sum: f64 = set.log_q.iter().map(|v| v.exp()).sum();
        approx_eq(sum, 1.0);
    }

    // ── Semi-adapted: 1 root, outlier active, constant scorer ────────────────────
    // With 1 root + outlier: existing_or_outlier = [Outlier, ExistingNode{r0}]
    //   normalized scores (both 0.0) = [-log(2), -log(2)]
    //   → each gets log(0.5) - log(2) = -log(4)
    // NewNode{k} for roots=1: k∈{0,1}
    //   k=0: semi_new_log_q(1,0) = log(0.5) - log(2) - 0 = log(0.5) - log(2)
    //   k=1: semi_new_log_q(1,1) = log(0.5) - log(2) - 0 = log(0.5) - log(2)
    // All 4 candidates → log_q = log(0.5) - log(2) = -log(4), sum = 1.0
    #[test]
    fn oracle_semi_adapted_one_root_outlier_active() {
        let tree = make_n_root_tree(1); // root: r0
        let adder = CompatTreeShellNodeAdder::default();

        let set =
            build_semi_adapted_proposal_set(&tree, &adder, 99, &[vec![0.0; 3]], true, const_scorer)
                .expect("should build");

        // Outlier + ExistingNode{r0} + NewNode{[]} + NewNode{[r0]} = 4
        assert_eq!(set.candidates.len(), 4);

        let expected = 0.5_f64.ln() - 2_f64.ln(); // log(0.25)
        for (i, q) in set.log_q.iter().enumerate() {
            approx_eq_strict(*q, expected, &format!("log_q[{}]", i));
        }

        let sum: f64 = set.log_q.iter().map(|v| v.exp()).sum();
        approx_eq(sum, 1.0);
    }

    // ── Fully-adapted: 2 roots, outlier inactive, constant scorer ────────────────
    // With constant scorer every candidate gets the same score 0.0.
    // Normalized: each = -log(N) where N = total candidates.
    // Candidates: 4 new-node + 2 existing = 6.
    #[test]
    fn oracle_fully_adapted_two_roots_outlier_inactive() {
        let tree = make_n_root_tree(2);
        let adder = CompatTreeShellNodeAdder::default();

        let set = build_fully_adapted_proposal_set(
            &tree,
            &adder,
            99,
            &[vec![0.0; 3]],
            false,
            const_scorer,
        )
        .expect("should build");

        // C(2,0)+C(2,1)+C(2,2) new-node + 2 existing = 4 + 2 = 6
        assert_eq!(set.candidates.len(), 6);

        let expected = -(6_f64.ln()); // -log(6)
        for (i, q) in set.log_q.iter().enumerate() {
            approx_eq_strict(*q, expected, &format!("log_q[{}]", i));
        }

        let sum: f64 = set.log_q.iter().map(|v| v.exp()).sum();
        approx_eq(sum, 1.0);
    }

    // ── Fully-adapted: 2 roots, outlier active, constant scorer ─────────────────
    // Candidates: 4 new-node + 1 outlier + 2 existing = 7.
    #[test]
    fn oracle_fully_adapted_two_roots_outlier_active() {
        let tree = make_n_root_tree(2);
        let adder = CompatTreeShellNodeAdder::default();

        let set = build_fully_adapted_proposal_set(
            &tree,
            &adder,
            99,
            &[vec![0.0; 3]],
            true,
            const_scorer,
        )
        .expect("should build");

        assert_eq!(set.candidates.len(), 7);

        let expected = -(7_f64.ln());
        for (i, q) in set.log_q.iter().enumerate() {
            approx_eq_strict(*q, expected, &format!("log_q[{}]", i));
        }

        let sum: f64 = set.log_q.iter().map(|v| v.exp()).sum();
        approx_eq(sum, 1.0);
    }

    // ── Bootstrap: 2 roots, outlier inactive, non-first particle ────────────────
    // Formulas (PhyClone bootstrap, no outlier):
    //   ExistingNode: log(0.5) - log(num_roots)
    //   NewNode{k}:   log(0.5) - log(num_roots+1) - log(C(num_roots,k))
    #[test]
    fn oracle_bootstrap_two_roots_outlier_inactive() {
        let tree = make_n_root_tree(2);
        let adder = CompatTreeShellNodeAdder::default();

        let set = build_bootstrap_proposal_set(&tree, &adder, 99, &[vec![0.0; 3]], false, false)
            .expect("should build");

        // 2 existing + 4 new-node = 6 candidates
        assert_eq!(set.candidates.len(), 6);

        let lh = 0.5_f64.ln(); // log(0.5)

        // ExistingNode: log(0.5) - log(2)
        approx_eq_strict(set.log_q[0], lh - 2_f64.ln(), "existing r0");
        approx_eq_strict(set.log_q[1], lh - 2_f64.ln(), "existing r1");

        // NewNode combos in order [], [r0], [r1], [r0,r1]
        approx_eq_strict(set.log_q[2], lh - 3_f64.ln(), "NewNode{{[]}}");
        approx_eq_strict(
            set.log_q[3],
            lh - 3_f64.ln() - 2_f64.ln(),
            "NewNode{{[r0]}}",
        );
        approx_eq_strict(
            set.log_q[4],
            lh - 3_f64.ln() - 2_f64.ln(),
            "NewNode{{[r1]}}",
        );
        approx_eq_strict(set.log_q[5], lh - 3_f64.ln(), "NewNode{{[r0,r1]}}");

        let sum: f64 = set.log_q.iter().map(|v| v.exp()).sum();
        approx_eq(sum, 1.0);
    }

    // ── Bootstrap: 2 roots, outlier active (outlier_prob=0.1), non-first ─────────
    // ExistingNode: log(0.45) - log(2)
    // NewNode{k}:   log(0.45) - log(3) - log(C(2,k))
    // Outlier:      log(0.1)
    #[test]
    fn oracle_bootstrap_two_roots_outlier_active() {
        let tree = make_n_root_tree(2);
        let adder = CompatTreeShellNodeAdder::default();

        let set = build_bootstrap_proposal_set(&tree, &adder, 99, &[vec![0.0; 3]], true, false)
            .expect("should build");

        // 2 existing + 1 outlier + 4 new-node = 7
        assert_eq!(set.candidates.len(), 7);

        // bootstrap ordering: outlier first, then existing, then new-node
        let log045 = 0.45_f64.ln();
        let log01 = 0.1_f64.ln();

        // index 0 = Outlier
        assert!(
            matches!(&set.candidates[0].kind, CompatProposalKind::Outlier),
            "candidates[0] should be Outlier"
        );
        approx_eq_strict(set.log_q[0], log01, "Outlier");

        // indices 1,2 = ExistingNode{r0}, ExistingNode{r1}
        approx_eq_strict(set.log_q[1], log045 - 2_f64.ln(), "existing r0");
        approx_eq_strict(set.log_q[2], log045 - 2_f64.ln(), "existing r1");

        // indices 3..6 = new-node combos
        approx_eq_strict(set.log_q[3], log045 - 3_f64.ln(), "NewNode{{[]}}");
        approx_eq_strict(
            set.log_q[4],
            log045 - 3_f64.ln() - 2_f64.ln(),
            "NewNode{{[r0]}}",
        );
        approx_eq_strict(
            set.log_q[5],
            log045 - 3_f64.ln() - 2_f64.ln(),
            "NewNode{{[r1]}}",
        );
        approx_eq_strict(set.log_q[6], log045 - 3_f64.ln(), "NewNode{{[r0,r1]}}");

        let sum: f64 = set.log_q.iter().map(|v| v.exp()).sum();
        approx_eq(sum, 1.0);
    }

    // ── Cross-kernel: 3 roots + outlier active, all kernels sum to 1.0 ───────────
    // Verifies that every proposal kernel yields a valid probability distribution
    // for the same tree, regardless of scorer.
    #[test]
    fn oracle_all_kernels_sum_to_one_three_roots_outlier_active() {
        let tree = make_n_root_tree(3);
        let adder = CompatTreeShellNodeAdder::default();

        let boot = build_bootstrap_proposal_set(&tree, &adder, 99, &[vec![0.0; 3]], true, false)
            .expect("bootstrap");
        let full = build_fully_adapted_proposal_set(
            &tree,
            &adder,
            99,
            &[vec![0.0; 3]],
            true,
            const_scorer,
        )
        .expect("fully adapted");
        let semi =
            build_semi_adapted_proposal_set(&tree, &adder, 99, &[vec![0.0; 3]], true, const_scorer)
                .expect("semi adapted");

        for (name, set) in [
            ("bootstrap", &boot),
            ("fully_adapted", &full),
            ("semi_adapted", &semi),
        ] {
            let sum: f64 = set.log_q.iter().map(|v| v.exp()).sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "{}: prob sum = {} (expected 1.0)",
                name,
                sum
            );
        }
    }
}
