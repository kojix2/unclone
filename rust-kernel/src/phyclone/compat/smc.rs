#![allow(dead_code)]

use super::data::CompatDataPoint;
use super::distributions::CompatTreeJointDistribution;
use super::proposal::{
    build_bootstrap_proposal_set, build_fully_adapted_proposal_set,
    build_semi_adapted_proposal_set, CompatProposalSet, CompatTreeShellNodeAdder,
};
use super::root_permutation::RootPermutationDistribution;
use super::tree_ids::{DataPointId, NodeId};
use super::tree_model::CompatTree;
use super::tree_stats::CompatOutlierPoint;
use rand::{Rng, RngExt};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

const SMC_ERR_DEGENERATE_WEIGHTS: &str = "[smc:degenerate_weights]";
const SMC_ERR_NO_CONSTRAINED_PROPOSAL: &str = "[smc:no_constrained_proposal_matched]";
const SMC_ERR_INVALID_CONSTRAINED_PATH: &str = "[smc:invalid_constrained_path]";

#[cfg(test)]
static SUBTREE_PG_CORE_PATH_HITS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
fn subtree_pg_record_core_path_hit() {
    SUBTREE_PG_CORE_PATH_HITS.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
fn subtree_pg_reset_core_path_hits() {
    SUBTREE_PG_CORE_PATH_HITS.store(0, Ordering::Relaxed);
}

#[cfg(test)]
fn subtree_pg_core_path_hits() -> usize {
    SUBTREE_PG_CORE_PATH_HITS.load(Ordering::Relaxed)
}

/// Proposal family selector, mirroring PhyClone's `proposal` argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ProposalFamily {
    Bootstrap,
    FullyAdapted,
    #[default]
    SemiAdapted,
}

impl ProposalFamily {
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => ProposalFamily::Bootstrap,
            1 => ProposalFamily::FullyAdapted,
            _ => ProposalFamily::SemiAdapted,
        }
    }
}

/// Typed SMC failures used to decide whether a PG step can be treated as a reject.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SmcError {
    DegenerateWeights,
    NoConstrainedProposalMatched,
    InvalidConstrainedPath,
    Other(String),
}

impl SmcError {
    pub fn from_message(message: &str) -> Self {
        if message.starts_with(SMC_ERR_DEGENERATE_WEIGHTS) {
            SmcError::DegenerateWeights
        } else if message.starts_with(SMC_ERR_NO_CONSTRAINED_PROPOSAL) {
            SmcError::NoConstrainedProposalMatched
        } else if message.starts_with(SMC_ERR_INVALID_CONSTRAINED_PATH) {
            SmcError::InvalidConstrainedPath
        } else {
            SmcError::Other(message.to_string())
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompatTreeHolder {
    pub tree: CompatTree,
}

impl CompatTreeHolder {
    pub fn new(tree: CompatTree) -> Self {
        Self { tree }
    }
}

#[derive(Clone, Debug)]
pub struct CompatParticle {
    pub holder: CompatTreeHolder,
    pub log_weight: f64,
    /// Current joint log-probability of the tree (log p). Populated during SMC.
    pub log_p: f64,
    /// Current normalised log-probability of the tree (log p_one). Populated during SMC.
    pub log_p_one: f64,
    /// Index of the parent particle in the parent swarm. Used for conditional SMC ancestry chain.
    pub parent_idx: Option<usize>,
}

impl CompatParticle {
    pub fn new(tree: CompatTree, log_weight: f64) -> Self {
        Self {
            holder: CompatTreeHolder::new(tree),
            log_weight,
            log_p: 0.0,
            log_p_one: 0.0,
            parent_idx: None,
        }
    }

    pub fn new_with_log_p(tree: CompatTree, log_weight: f64, log_p: f64, log_p_one: f64) -> Self {
        Self {
            holder: CompatTreeHolder::new(tree),
            log_weight,
            log_p,
            log_p_one,
            parent_idx: None,
        }
    }

    pub fn new_with_ancestry(
        tree: CompatTree,
        log_weight: f64,
        log_p: f64,
        log_p_one: f64,
        parent_idx: Option<usize>,
    ) -> Self {
        Self {
            holder: CompatTreeHolder::new(tree),
            log_weight,
            log_p,
            log_p_one,
            parent_idx,
        }
    }

    pub fn update_from_proposal(
        &mut self,
        proposal_set: &CompatProposalSet,
        proposal_index: usize,
        proposal_log_target: f64,
    ) -> Result<(), String> {
        let candidate = proposal_set
            .candidates
            .get(proposal_index)
            .ok_or_else(|| "proposal index out of bounds".to_string())?;
        let log_q = proposal_set
            .log_q
            .get(proposal_index)
            .copied()
            .ok_or_else(|| "log_q index out of bounds".to_string())?;

        if !log_q.is_finite() || !proposal_log_target.is_finite() {
            return Err("proposal_log_target and log_q must be finite".to_string());
        }

        self.holder.tree = candidate.tree.clone();
        self.log_weight += proposal_log_target - log_q;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct CompatParticleSwarm {
    pub particles: Vec<CompatParticle>,
}

impl CompatParticleSwarm {
    pub fn new(particles: Vec<CompatParticle>) -> Self {
        Self { particles }
    }

    pub fn normalize_log_weights(&mut self) -> Result<(), String> {
        if self.particles.is_empty() {
            return Err("cannot normalize empty particle swarm".to_string());
        }

        let max = self
            .particles
            .iter()
            .map(|p| p.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);

        if max.is_infinite() && max.is_sign_negative() {
            return Err(format!(
                "{} all particle log weights are -inf",
                SMC_ERR_DEGENERATE_WEIGHTS
            ));
        }

        let sum_exp: f64 = self
            .particles
            .iter()
            .map(|p| (p.log_weight - max).exp())
            .sum();
        let lse = max + sum_exp.ln();

        for particle in &mut self.particles {
            particle.log_weight -= lse;
        }

        Ok(())
    }

    pub fn effective_sample_size(&self) -> Result<f64, String> {
        if self.particles.is_empty() {
            return Err("cannot compute ESS for empty particle swarm".to_string());
        }

        let probs: Vec<f64> = self.particles.iter().map(|p| p.log_weight.exp()).collect();
        if probs.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err("particle weights must be finite and non-negative".to_string());
        }

        let sum: f64 = probs.iter().sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err("sum of particle weights must be positive".to_string());
        }

        let sum_sq: f64 = probs
            .iter()
            .map(|w| {
                let p = *w / sum;
                p * p
            })
            .sum();

        if sum_sq <= 0.0 || !sum_sq.is_finite() {
            return Err("invalid squared-weight sum for ESS".to_string());
        }

        Ok(1.0 / sum_sq)
    }

    /// ESS relative to the number of particles (0..1).
    pub fn relative_ess(&self) -> Result<f64, String> {
        let ess = self.effective_sample_size()?;
        Ok(ess / self.particles.len() as f64)
    }

    /// Multinomial resampling matching PhyClone's `rng.multinomial(num_particles, weights)`.
    /// Each of the `n` new particles is drawn independently from the categorical distribution
    /// over the current swarm. After resampling all particles carry uniform log weight
    /// `-log(n)`.
    pub fn resample(&mut self, rng: &mut impl Rng) -> Result<(), String> {
        let n = self.particles.len();
        if n == 0 {
            return Err("cannot resample empty swarm".to_string());
        }

        // Convert log weights to normalized probabilities.
        let max = self
            .particles
            .iter()
            .map(|p| p.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut probs: Vec<f64> = self
            .particles
            .iter()
            .map(|p| (p.log_weight - max).exp())
            .collect();

        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Err("invalid probabilities for resampling".to_string());
        }
        for p in &mut probs {
            *p /= sum;
        }

        // Build CDF for multinomial sampling.
        let mut cdf = Vec::with_capacity(n);
        let mut cumulative = 0.0_f64;
        for &prob in &probs {
            cumulative += prob;
            cdf.push(cumulative);
        }

        let uniform_log_weight = -(n as f64).ln();
        let mut new_particles = Vec::with_capacity(n);

        for _ in 0..n {
            let u: f64 = rng.random();
            let source_idx = cdf.partition_point(|&c| c < u).min(n - 1);
            let mut p = self.particles[source_idx].clone();
            p.log_weight = uniform_log_weight;
            p.parent_idx = Some(source_idx);
            new_particles.push(p);
        }

        self.particles = new_particles;
        Ok(())
    }

    /// Conditional (Particle Gibbs) resampling: particles[0] is the retained path and
    /// is always copied as-is into slot 0.  The remaining N-1 slots are filled by
    /// multinomial resampling from ALL N particles (including particle 0), which
    /// maintains the correct conditional distribution.
    ///
    /// After resampling every particle carries uniform log weight `-log(N)`.
    /// parent_idx is set to track ancestry chains for constrained path reconstruction.
    pub fn resample_conditional(&mut self, rng: &mut impl Rng) -> Result<(), String> {
        let n = self.particles.len();
        if n < 2 {
            return Err("conditional resampling requires at least 2 particles".to_string());
        }

        let max = self
            .particles
            .iter()
            .map(|p| p.log_weight)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut probs: Vec<f64> = self
            .particles
            .iter()
            .map(|p| (p.log_weight - max).exp())
            .collect();

        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return Err("invalid probabilities for conditional resampling".to_string());
        }
        for p in &mut probs {
            *p /= sum;
        }

        let uniform_log_weight = -(n as f64).ln();

        // Slot 0: always the retained particle (parent_idx = 0, i.e., self-loop).
        let mut retained = self.particles[0].clone();
        retained.log_weight = uniform_log_weight;
        retained.parent_idx = Some(0);

        // Build CDF for multinomial sampling of remaining N-1 slots from all N particles.
        let mut cdf = Vec::with_capacity(n);
        let mut cumulative = 0.0_f64;
        for &prob in &probs {
            cumulative += prob;
            cdf.push(cumulative);
        }

        let mut new_particles = Vec::with_capacity(n);
        new_particles.push(retained);

        for _ in 0..(n - 1) {
            let u: f64 = rng.random();
            let source_idx = cdf.partition_point(|&c| c < u).min(n - 1);
            let mut p = self.particles[source_idx].clone();
            p.log_weight = uniform_log_weight;
            p.parent_idx = Some(source_idx);
            new_particles.push(p);
        }

        self.particles = new_particles;
        Ok(())
    }

    /// Extract the ancestry chain for particle at `leaf_idx` by following parent_idx backwards.
    /// Returns a Vec of particle indices from root (earliest ancestor) to leaf.
    ///
    /// Example: if ancestry chain is A <- B <- C (where C is leaf):
    /// `get_ancestry_chain(C_idx)` returns `[A_idx, B_idx, C_idx]`.
    pub fn get_ancestry_chain(&self, leaf_idx: usize) -> Result<Vec<usize>, String> {
        if leaf_idx >= self.particles.len() {
            return Err(format!(
                "leaf_idx {} out of bounds for swarm of size {}",
                leaf_idx,
                self.particles.len()
            ));
        }

        let mut chain = vec![leaf_idx];
        let mut current = leaf_idx;
        let max_depth = self.particles.len() + 10; // Prevent infinite loops

        for _ in 0..max_depth {
            let Some(parent_idx) = self.particles[current].parent_idx else {
                // Reached root (no parent)
                break;
            };

            if parent_idx == current {
                // Self-loop marks retained particle in conditional resampling
                break;
            }

            chain.push(parent_idx);
            current = parent_idx;
        }

        chain.reverse();
        Ok(chain)
    }
}

/// Sample an index from `log_weights` proportional to `exp(log_weights[i])`.
/// Returns an error if the slice is empty or all weights are -inf.
pub fn sample_discrete_from_log_weights(
    log_weights: &[f64],
    rng: &mut impl Rng,
) -> Result<usize, String> {
    if log_weights.is_empty() {
        return Err("log_weights must not be empty".to_string());
    }

    let max = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    if max.is_infinite() && max.is_sign_negative() {
        return Err("all log_weights are -inf".to_string());
    }

    let weights: Vec<f64> = log_weights.iter().map(|&w| (w - max).exp()).collect();
    let sum: f64 = weights.iter().sum();

    let threshold = rng.random::<f64>() * sum;
    let mut cumulative = 0.0_f64;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w;
        if cumulative >= threshold {
            return Ok(i);
        }
    }

    // Fallback: return last index (handles floating-point edge cases).
    Ok(log_weights.len() - 1)
}

/// Re-order `data_points` according to the index sequence in `data_sigma`.
///
/// `data_sigma` is a list of `DataPointId` values (the permutation produced by
/// `RootPermutationDistribution::sample`). Returns a new Vec whose order matches
/// `data_sigma`. Data point IDs that appear in `data_sigma` but not in `data_points`
/// are silently skipped (e.g. outliers are excluded from the SMC sequence).
pub fn reorder_data_points(
    data_points: &[CompatDataPoint],
    data_sigma: &[usize],
) -> Result<Vec<CompatDataPoint>, String> {
    use std::collections::HashMap;
    let index: HashMap<usize, &CompatDataPoint> =
        data_points.iter().map(|dp| (dp.idx, dp)).collect();
    let mut ordered = Vec::with_capacity(data_sigma.len());
    for &dp_id in data_sigma {
        if let Some(dp) = index.get(&dp_id) {
            ordered.push((*dp).clone());
        }
        // Outliers are in data_sigma but not in the non-outlier data_points slice;
        // they are skipped here (SMC only processes non-outlier data points).
    }
    if ordered.is_empty() && !data_points.is_empty() {
        return Err(
            "reorder_data_points: data_sigma contained no ids matching data_points".to_string(),
        );
    }
    Ok(ordered)
}

/// One step in the constrained particle sequence built by [`build_constrained_particle_sequence`].
#[derive(Clone, Debug)]
pub struct ConstrainedStep {
    /// Constrained tree state after adding the data point at this step.
    pub tree: CompatTree,
    /// Log joint probability of this tree (`log_p`).
    pub log_p: f64,
    /// Normalised log probability (`log_p_one`), used at the final SMC step.
    pub log_p_one: f64,
    /// Log proposal probability for this step.
    pub log_q: f64,
}

/// Build the constrained particle sequence from a retained tree.
///
/// Mirrors PhyClone's `ConditionalSMCSampler._get_constrained_path`.
///
/// For each data point in `data_points` (in sigma order) this function:
/// 1. Determines the action (outlier / existing-node / new-node-with-children)
///    by consulting the retained tree's topology and the running `node_map`.
/// 2. Builds the proposal set from the current intermediate tree state.
/// 3. Identifies the matching candidate by **action kind**, not by final node id.
/// 4. Adopts the candidate's tree as the next intermediate tree.
/// 5. Records `log_p`, `log_p_one`, and `log_q` for weight computation.
///
/// The sigma order produced by `RootPermutationDistribution` guarantees that children
/// of any node in the retained tree are processed before the node itself (the first
/// data point from a child node creates that node, so all children exist in `node_map`
/// by the time the parent node's first data point is encountered).
pub fn build_constrained_particle_sequence(
    retained_tree: &CompatTree,
    data_points: &[CompatDataPoint],
    joint: &CompatTreeJointDistribution,
    proposal: ProposalFamily,
) -> Result<Vec<ConstrainedStep>, String> {
    use super::proposal::CompatProposalKind;
    use std::collections::HashMap;

    if data_points.is_empty() {
        return Err("data_points must not be empty".to_string());
    }

    let data_to_node = retained_tree.data_point_to_node_map();
    const OUTLIER_SENTINEL: &str = "-1";

    // Maps retained-tree node_id -> new_tree node_id as intermediate trees grow.
    let mut node_map: HashMap<String, String> = HashMap::new();

    let first_dp = &data_points[0];
    let num_samples = first_dp.value.len();
    let num_grid_points = first_dp.value.first().map(|row| row.len()).unwrap_or(0);
    let log_prior = if num_grid_points > 0 {
        -(num_grid_points as f64).ln()
    } else {
        0.0
    };
    let cache_entries = (data_points.len() + 1).next_power_of_two().max(8);

    let adder = CompatTreeShellNodeAdder::default();
    let outlier_points = CompatTree::outlier_points(data_points);
    let scorer = |tree: &CompatTree| -> f64 {
        joint
            .log_p_tree(tree, &outlier_points)
            .unwrap_or(f64::NEG_INFINITY)
    };

    let mut new_tree = CompatTree::new(
        "root",
        num_samples,
        num_grid_points,
        log_prior,
        cache_entries,
    );

    let mut steps: Vec<ConstrainedStep> = Vec::with_capacity(data_points.len());

    for dp in data_points {
        let dp_id = dp.idx;
        let dp_value = &dp.value;

        // Determine which node in the retained tree this data point belongs to.
        let old_node_id: &str = data_to_node.get(&dp_id).copied().ok_or_else(|| {
            format!(
                "build_constrained_particle_sequence: dp_id={} not in retained tree",
                dp_id
            )
        })?;

        // Map the retained-tree action to a new-tree action.
        let action_kind: CompatProposalKind = if old_node_id == OUTLIER_SENTINEL {
            CompatProposalKind::Outlier
        } else if let Some(new_node_id) = node_map.get(old_node_id) {
            // Node already created in new_tree – add to existing node.
            CompatProposalKind::ExistingNode {
                node_id: new_node_id.clone(),
            }
        } else {
            // First data point for this retained-tree node: must create a new node.
            // Collect the children of old_node (all must already be in node_map due to
            // the bottom-up ordering guaranteed by RootPermutationDistribution).
            let old_children = retained_tree.children_of(old_node_id)?;
            let new_children: Vec<String> = old_children
                .iter()
                .map(|&child_old| {
                    node_map.get(child_old).cloned().ok_or_else(|| {
                        format!(
                            "build_constrained_particle_sequence: child {} of {} \
                             not yet in node_map (sigma order violation?)",
                            child_old, old_node_id
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            CompatProposalKind::NewNode {
                children: new_children,
            }
        };

        // Build proposal set from the current intermediate tree.
        let proposal_set = build_proposal_set_for_kind(
            proposal,
            &new_tree,
            &adder,
            dp_id,
            dp_value,
            joint.outlier_modelling_active,
            steps.is_empty(),
            scorer,
        )?;

        if proposal_set.is_empty() {
            return Err(format!(
                "build_constrained_particle_sequence: empty proposal set at dp_id={}",
                dp_id
            ));
        }

        // Find the candidate whose action kind matches (by structure, not by node id).
        let candidate_idx = find_constrained_candidate_by_kind(&proposal_set, &action_kind)
            .ok_or_else(|| {
                format!(
                    "{} dp_id={} action={:?}",
                    SMC_ERR_NO_CONSTRAINED_PROPOSAL, dp_id, action_kind
                )
            })?;

        let log_q = proposal_set.log_q[candidate_idx];
        let candidate_tree = proposal_set.candidates[candidate_idx].tree.clone();

        // Update node_map when a new node was created.
        if let CompatProposalKind::NewNode { .. } = &action_kind {
            let new_node_id = candidate_tree
                .node_id_for_data_point(dp_id)
                .ok_or_else(|| {
                    format!(
                        "build_constrained_particle_sequence: dp_id={} not assigned in candidate",
                        dp_id
                    )
                })?
                .to_string();
            node_map.insert(old_node_id.to_string(), new_node_id);
        }

        // Advance the intermediate tree to the candidate's tree.
        new_tree = candidate_tree;

        let (log_p, log_p_one) =
            joint.compute_log_p_and_log_p_one_tree(&new_tree, &outlier_points)?;

        steps.push(ConstrainedStep {
            tree: new_tree.clone(),
            log_p,
            log_p_one,
            log_q,
        });
    }

    Ok(steps)
}

/// Find the index of the first candidate in `proposal_set` whose `kind` matches
/// `action_kind` structurally.
///
/// For `NewNode`, matching is set-equality of children (order-independent).
fn find_constrained_candidate_by_kind(
    proposal_set: &CompatProposalSet,
    action_kind: &super::proposal::CompatProposalKind,
) -> Option<usize> {
    use super::proposal::CompatProposalKind;

    for (idx, candidate) in proposal_set.candidates.iter().enumerate() {
        let matched = match (action_kind, &candidate.kind) {
            (CompatProposalKind::Outlier, CompatProposalKind::Outlier) => true,
            (
                CompatProposalKind::ExistingNode { node_id: target },
                CompatProposalKind::ExistingNode { node_id: cand },
            ) => target == cand,
            (
                CompatProposalKind::NewNode {
                    children: target_ch,
                },
                CompatProposalKind::NewNode { children: cand_ch },
            ) => {
                if target_ch.len() != cand_ch.len() {
                    false
                } else {
                    let mut tc = target_ch.clone();
                    let mut cc = cand_ch.clone();
                    tc.sort();
                    cc.sort();
                    tc == cc
                }
            }
            _ => false,
        };
        if matched {
            return Some(idx);
        }
    }
    None
}

/// Build a proposal set using the selected `ProposalFamily`.
#[allow(clippy::too_many_arguments)]
fn build_proposal_set_for_kind<F>(
    kind: ProposalFamily,
    tree: &CompatTree,
    adder: &CompatTreeShellNodeAdder,
    dp_id: usize,
    dp_value: &[Vec<f64>],
    outlier_modelling_active: bool,
    is_initial_step: bool,
    scorer: F,
) -> Result<CompatProposalSet, String>
where
    F: Fn(&CompatTree) -> f64,
{
    match kind {
        ProposalFamily::Bootstrap => build_bootstrap_proposal_set(
            tree,
            adder,
            dp_id,
            dp_value,
            outlier_modelling_active,
            is_initial_step,
        ),
        ProposalFamily::FullyAdapted => build_fully_adapted_proposal_set(
            tree,
            adder,
            dp_id,
            dp_value,
            outlier_modelling_active,
            scorer,
        ),
        ProposalFamily::SemiAdapted => build_semi_adapted_proposal_set(
            tree,
            adder,
            dp_id,
            dp_value,
            outlier_modelling_active,
            scorer,
        ),
    }
}

/// Run a single-pass conditional SMC filter over `data_points` (already in sigma order),
/// retaining a trajectory consistent with `retained_tree` at particle slot 0.
///
/// Returns the final particle swarm with **unnormalized** log weights (i.e., the raw
/// weights from the SMC loop, before normalization).  The caller is responsible for
/// normalizing and sampling.
///
/// This lower-level function is used by `run_conditional_smc` (which also normalizes
/// and samples) and by the subtree Particle Gibbs sampler (which corrects weights
/// before sampling).
pub fn run_conditional_smc_swarm(
    joint: &CompatTreeJointDistribution,
    data_points: &[CompatDataPoint],
    retained_tree: &CompatTree,
    num_particles: usize,
    resample_threshold: f64,
    proposal: ProposalFamily,
    rng: &mut impl Rng,
) -> Result<CompatParticleSwarm, String> {
    if data_points.is_empty() {
        return Err("data_points must not be empty".to_string());
    }
    if num_particles < 2 {
        return Err("conditional SMC requires num_particles >= 2".to_string());
    }

    // Pre-build the full constrained particle sequence (PhyClone _get_constrained_path).
    let constrained_steps =
        build_constrained_particle_sequence(retained_tree, data_points, joint, proposal)?;

    let num_dps = data_points.len();
    let outlier_points = CompatTree::outlier_points(data_points);
    let scorer = |tree: &CompatTree| -> f64 {
        joint
            .log_p_tree(tree, &outlier_points)
            .unwrap_or(f64::NEG_INFINITY)
    };

    let adder = CompatTreeShellNodeAdder::default();
    let first_dp = &data_points[0];
    let num_samples = first_dp.value.len();
    let num_grid_points = first_dp.value.first().map(|row| row.len()).unwrap_or(0);
    let log_prior = if num_grid_points > 0 {
        -(num_grid_points as f64).ln()
    } else {
        0.0
    };
    let base_tree = CompatTree::new("root", num_samples, num_grid_points, log_prior, 8);
    let uniform_log_weight = -(num_particles as f64).ln();

    // ── Step 0 initialisation ──────────────────────────────────────────────────
    let dp0 = &data_points[0];
    let mut init_particles: Vec<CompatParticle> = Vec::with_capacity(num_particles);

    init_particles.push(CompatParticle::new_with_log_p(
        constrained_steps[0].tree.clone(),
        uniform_log_weight,
        constrained_steps[0].log_p,
        constrained_steps[0].log_p_one,
    ));

    for _ in 0..(num_particles - 1) {
        let proposal_set = build_proposal_set_for_kind(
            proposal,
            &base_tree,
            &adder,
            dp0.idx,
            &dp0.value,
            joint.outlier_modelling_active,
            true,
            scorer,
        )?;
        if proposal_set.is_empty() {
            return Err(format!(
                "run_conditional_smc_swarm: empty proposal set at step 0 for dp_id={}",
                dp0.idx
            ));
        }
        let idx = sample_discrete_from_log_weights(&proposal_set.log_q, rng)?;
        let new_tree = proposal_set.candidates[idx].tree.clone();
        let (new_log_p, new_log_p_one) =
            joint.compute_log_p_and_log_p_one_tree(&new_tree, &outlier_points)?;
        init_particles.push(CompatParticle::new_with_log_p(
            new_tree,
            uniform_log_weight,
            new_log_p,
            new_log_p_one,
        ));
    }

    let mut swarm = CompatParticleSwarm::new(init_particles);

    // ── Steps 1 .. T-1 ────────────────────────────────────────────────────────
    for step in 1..num_dps {
        let is_last = step == num_dps - 1;
        let constrained = &constrained_steps[step];
        let prev_constrained_log_p = constrained_steps[step - 1].log_p;

        let rel_ess = swarm.relative_ess()?;
        if rel_ess <= resample_threshold {
            swarm.resample_conditional(rng)?;
        }

        let dp = &data_points[step];
        let dp_id = dp.idx;
        let dp_value = &dp.value;

        let mut new_particles: Vec<CompatParticle> = Vec::with_capacity(num_particles);

        for (particle_idx, parent) in swarm.particles.iter().enumerate() {
            let parent_log_weight = parent.log_weight;
            let parent_log_p = parent.log_p;

            if particle_idx == 0 {
                let log_target = if is_last {
                    constrained.log_p_one
                } else {
                    constrained.log_p
                };
                let log_w_delta = log_target - prev_constrained_log_p - constrained.log_q;
                new_particles.push(CompatParticle::new_with_ancestry(
                    constrained.tree.clone(),
                    parent_log_weight + log_w_delta,
                    constrained.log_p,
                    constrained.log_p_one,
                    Some(0),
                ));
            } else {
                let proposal_set = build_proposal_set_for_kind(
                    proposal,
                    &parent.holder.tree,
                    &adder,
                    dp_id,
                    dp_value,
                    joint.outlier_modelling_active,
                    false,
                    scorer,
                )?;
                if proposal_set.is_empty() {
                    return Err(format!(
                        "run_conditional_smc_swarm: empty proposal set at step {} for dp_id={}",
                        step, dp_id
                    ));
                }
                let proposal_idx = sample_discrete_from_log_weights(&proposal_set.log_q, rng)?;
                let log_q = proposal_set.log_q[proposal_idx];
                let new_tree = proposal_set.candidates[proposal_idx].tree.clone();
                let (new_log_p, new_log_p_one) =
                    joint.compute_log_p_and_log_p_one_tree(&new_tree, &outlier_points)?;
                let log_target = if is_last { new_log_p_one } else { new_log_p };
                let delta = log_target - parent_log_p - log_q;
                new_particles.push(CompatParticle::new_with_ancestry(
                    new_tree,
                    parent_log_weight + delta,
                    new_log_p,
                    new_log_p_one,
                    Some(particle_idx),
                ));
            }
        }

        swarm.particles = new_particles;
    }

    Ok(swarm)
}

/// Sample a tree from an **unnormalized** particle swarm.
///
/// This helper normalizes the log weights and then draws one particle
/// proportional to the normalized probabilities.
pub fn sample_tree_from_swarm(
    swarm: &mut CompatParticleSwarm,
    rng: &mut impl Rng,
) -> Result<CompatTree, String> {
    swarm.normalize_log_weights()?;
    let sampled_idx = sample_discrete_from_log_weights(
        &swarm
            .particles
            .iter()
            .map(|p| p.log_weight)
            .collect::<Vec<_>>(),
        rng,
    )?;
    Ok(swarm.particles[sampled_idx].holder.tree.clone())
}

/// Run a single-pass conditional SMC filter over `data_points` (already in sigma order),
/// retaining a trajectory consistent with `retained_tree` at particle slot 0.
///
/// Mirrors PhyClone's `ConditionalSMCSampler.sample`:
/// - Pre-builds the constrained particle sequence via [`build_constrained_particle_sequence`].
/// - Slot 0 always carries the pre-built constrained particle (never searched at runtime).
/// - Resampling preserves slot 0 (`resample_conditional`) to maintain the Particle Gibbs
///   invariant.
/// - Weight updates for slot 0 use the pre-computed `log_p` / `log_p_one` and `log_q`.
/// - All other particles are proposed from the standard proposal distribution.
pub fn run_conditional_smc(
    joint: &CompatTreeJointDistribution,
    data_points: &[CompatDataPoint],
    retained_tree: &CompatTree,
    num_particles: usize,
    resample_threshold: f64,
    proposal: ProposalFamily,
    rng: &mut impl Rng,
) -> Result<CompatTree, String> {
    let mut swarm = run_conditional_smc_swarm(
        joint,
        data_points,
        retained_tree,
        num_particles,
        resample_threshold,
        proposal,
        rng,
    )?;
    sample_tree_from_swarm(&mut swarm, rng)
}

#[derive(Clone, Debug)]
pub struct CompatParticleGibbsSampler {
    pub joint: CompatTreeJointDistribution,
    pub num_particles: usize,
    pub resample_threshold: f64,
    pub proposal: ProposalFamily,
}

#[derive(Clone, Debug)]
pub struct CompatMcmcSample {
    pub iter: usize,
    pub tree: CompatTree,
    pub alpha: f64,
    pub log_p: f64,
    pub log_p_one: f64,
}

impl CompatParticleGibbsSampler {
    pub fn sample_tree(
        &self,
        data_points: &[CompatDataPoint],
        current_tree: &CompatTree,
        rng: &mut impl Rng,
    ) -> Result<CompatTree, String> {
        // Sample data order from current tree topology, mirroring PhyClone's
        // ParticleGibbsTreeSampler which calls RootPermutationDistribution.sample.
        let data_sigma = RootPermutationDistribution::sample(current_tree, rng);

        // Re-order data_points to match data_sigma.
        let ordered = reorder_data_points(data_points, &data_sigma)?;

        // Run conditional SMC: the constrained particle sequence is built internally
        // from `current_tree`, matching PhyClone's _get_constrained_path.
        run_conditional_smc(
            &self.joint,
            &ordered,
            current_tree,
            self.num_particles,
            self.resample_threshold,
            self.proposal,
            rng,
        )
    }
}

// ── Subtree Particle Gibbs Sampler ────────────────────────────────────────────

/// Particle Gibbs sampler that updates a randomly selected subtree rather than
/// the full tree.
///
/// Mirrors `ParticleGibbsSubtreeSampler` in `phyclone/mcmc/particle_gibbs.py`.
///
/// When the tree has ≤ 1 non-root nodes, or when the selected subtree root is
/// the tree root (direct child of root selected), the sampler falls back to a
/// full-tree Particle Gibbs step.
#[derive(Clone, Debug)]
pub struct CompatParticleGibbsSubtreeSampler {
    pub joint: CompatTreeJointDistribution,
    pub num_particles: usize,
    pub resample_threshold: f64,
    pub proposal: ProposalFamily,
}

impl CompatParticleGibbsSubtreeSampler {
    /// Sample a new tree by updating a randomly selected subtree via conditional SMC.
    pub fn sample_tree(
        &self,
        data_points: &[CompatDataPoint],
        current_tree: &CompatTree,
        rng: &mut impl Rng,
    ) -> Result<CompatTree, String> {
        // Guard: fall back to full-tree PG when the tree is too small.
        let non_root_count = current_tree.non_root_node_ids().count();
        if non_root_count <= 1 {
            return self.full_tree_pg(data_points, current_tree, rng);
        }

        // 1. Select subtree root child (data-point weighted, no outlier node).
        let subtree_root_child = match sample_subtree_root_child(current_tree, rng) {
            Some(c) => c,
            None => return self.full_tree_pg(data_points, current_tree, rng),
        };

        // 2. subtree_root = parent of subtree_root_child.
        let subtree_root = current_tree
            .parent_of(&subtree_root_child)?
            .map(|s| s.to_string())
            .unwrap_or_else(|| current_tree.root_node_id.clone());

        if subtree_root == current_tree.root_node_id {
            // Intentional approximation:
            // if the selected subtree is the whole tree, reuse the existing
            // full-tree Particle Gibbs kernel instead of reproducing PhyClone's
            // root-subtree extract/prune/graft/weight-correction path.
            // Strict parity would route this case through the same subtree
            // machinery used for non-root subtree updates.
            return self.full_tree_pg(data_points, current_tree, rng);
        }

        #[cfg(test)]
        subtree_pg_record_core_path_hit();

        // parent = the parent of subtree_root in the full tree (may be None for a
        // direct child of the tree root).
        let parent: Option<String> = current_tree
            .parent_of(&subtree_root)?
            .filter(|&p| p != current_tree.root_node_id.as_str())
            .map(|s| s.to_string());

        // 3. Extract the subtree and build a pruned copy of the full tree.
        let mut subtree = current_tree.extract_subtree_tree(&subtree_root)?;
        let mut pruned_tree = current_tree.clone();
        pruned_tree.remove_subtree_nodes(&subtree_root)?;

        // 4. Move outliers from the pruned tree into the subtree (matching PhyClone).
        let outlier_ids: Vec<DataPointId> = pruned_tree.assigned_outliers.iter().copied().collect();
        for dp_id in outlier_ids {
            pruned_tree.assigned_outliers.remove(&dp_id);
            subtree.assigned_outliers.insert(dp_id);
        }

        // 5. Collect the data points that belong to the subtree.
        let subtree_dp_ids = subtree.all_assigned_data_point_ids();
        let subtree_data_points = filter_data_points_by_ids(data_points, &subtree_dp_ids);
        if subtree_data_points.is_empty() {
            return Ok(current_tree.clone());
        }

        // 6. Run conditional SMC on the subtree.
        let data_sigma = RootPermutationDistribution::sample(&subtree, rng);
        let ordered_subtree = reorder_data_points(&subtree_data_points, &data_sigma)?;
        let mut swarm = run_conditional_smc_swarm(
            &self.joint,
            &ordered_subtree,
            &subtree,
            self.num_particles,
            self.resample_threshold,
            self.proposal,
            rng,
        )?;

        // 7. Correct particle weights so the target is the full-tree distribution.
        let outlier_points_full = CompatTree::outlier_points(data_points);
        self.correct_subtree_swarm_weights(
            &mut swarm,
            &pruned_tree,
            parent.as_deref(),
            &outlier_points_full,
        )?;

        // 8. Normalize and sample.
        sample_tree_from_swarm(&mut swarm, rng)
    }

    /// Correct subtree particle weights so each particle represents a full tree.
    ///
    /// For each particle:
    ///   w_corrected = w_subtree − log_p_one(subtree) + log_p_one(full tree after graft)
    ///
    /// Mirrors `_correct_weights` in `phyclone/mcmc/particle_gibbs.py`.
    fn correct_subtree_swarm_weights(
        &self,
        swarm: &mut CompatParticleSwarm,
        pruned_tree: &CompatTree,
        parent_id: Option<&str>,
        outlier_points: &[CompatOutlierPoint],
    ) -> Result<(), String> {
        for particle in &mut swarm.particles {
            let subtree_log_p_one = particle.log_p_one;

            // Graft this particle's proposed subtree onto the pruned full tree.
            let mut full_tree = pruned_tree.clone();
            full_tree.graft_subtree_tree(&particle.holder.tree, parent_id)?;

            // Refresh both scores so the particle remains internally consistent
            // after replacing the subtree proposal with the grafted full tree.
            let (full_log_p, full_log_p_one) = self
                .joint
                .compute_log_p_and_log_p_one_tree(&full_tree, outlier_points)?;

            // w_corrected = w_subtree − subtree_log_p_one + full_log_p_one
            particle.log_weight = particle.log_weight - subtree_log_p_one + full_log_p_one;
            particle.log_p = full_log_p;
            particle.log_p_one = full_log_p_one;
            particle.holder.tree = full_tree;
        }
        Ok(())
    }

    /// Full-tree Particle Gibbs fallback (delegates to `run_conditional_smc`).
    fn full_tree_pg(
        &self,
        data_points: &[CompatDataPoint],
        current_tree: &CompatTree,
        rng: &mut impl Rng,
    ) -> Result<CompatTree, String> {
        let data_sigma = RootPermutationDistribution::sample(current_tree, rng);
        let ordered = reorder_data_points(data_points, &data_sigma)?;
        run_conditional_smc(
            &self.joint,
            &ordered,
            current_tree,
            self.num_particles,
            self.resample_threshold,
            self.proposal,
            rng,
        )
    }
}

/// Select a subtree root child by sampling proportional to the number of
/// data points assigned to each non-root node.
///
/// Mirrors the data-point-weighted node selection in
/// `ParticleGibbsSubtreeSampler.sample_tree` (PhyClone).
fn sample_subtree_root_child(tree: &CompatTree, rng: &mut impl Rng) -> Option<NodeId> {
    let mut weighted: Vec<NodeId> = Vec::new();
    for node_id in tree.non_root_node_ids() {
        // Skip the outlier sentinel node.
        if node_id == "-1" {
            continue;
        }
        let count = tree.node(node_id).map_or(0, |n| n.data_point_ids.len());
        for _ in 0..count {
            weighted.push(node_id.to_string());
        }
    }
    if weighted.is_empty() {
        return None;
    }
    let idx = rng.random_range(0..weighted.len());
    Some(weighted[idx].clone())
}

/// Filter `all_data_points` to only those whose `idx` is in `ids`.
fn filter_data_points_by_ids(
    all_data_points: &[CompatDataPoint],
    ids: &[DataPointId],
) -> Vec<CompatDataPoint> {
    use std::collections::HashSet;
    let id_set: HashSet<DataPointId> = ids.iter().copied().collect();
    all_data_points
        .iter()
        .filter(|dp| id_set.contains(&dp.idx))
        .cloned()
        .collect()
}

/// Run a single-chain Particle Gibbs loop.
///
/// Returns `num_iters` sampled trees after burn-in and thinning.
/// Legacy implementation kept for backward compatibility.
pub fn run_compat_mcmc_inner(
    sampler: &CompatParticleGibbsSampler,
    data_points: &[CompatDataPoint],
    mut current_tree: CompatTree,
    burnin: usize,
    num_iters: usize,
    thin: usize,
    rng: &mut impl Rng,
) -> Result<Vec<CompatTree>, String> {
    if data_points.is_empty() {
        return Err("data_points must not be empty".to_string());
    }
    if thin == 0 {
        return Err("thin must be >= 1".to_string());
    }

    let post_burnin = num_iters.saturating_sub(burnin);
    let mut samples = Vec::with_capacity(post_burnin / thin);

    for i in 1..=num_iters {
        current_tree = sampler.sample_tree(data_points, &current_tree, rng)?;
        if i > burnin && (i - burnin).is_multiple_of(thin) {
            samples.push(current_tree.clone());
        }
    }

    Ok(samples)
}

// ── PhyClone-compatible MCMC ──────────────────────────────────────────────────

/// Configuration for a PhyClone-compatible MCMC chain.
#[derive(Clone, Debug)]
pub struct PhyCloneMcmcConfig {
    /// Number of burn-in iterations (UnconditionalSMC + DataPoint + PruneRegraph).
    pub burnin: usize,
    /// Number of post-burn-in iterations recorded.
    pub num_iters: usize,
    /// Maximum cumulative runtime in seconds across burn-in + main MCMC.
    pub max_time: f64,
    /// Progress print interval.
    pub print_freq: usize,
    /// Thinning factor (record every `thin`-th iteration).
    pub thin: usize,
    /// Number of Particle Gibbs particles.
    pub num_particles: usize,
    /// ESS resampling threshold (0..1).
    pub resample_threshold: f64,
    /// Gibbs passes for DataPointSampler per iteration.
    pub num_samples_data_point: usize,
    /// Gibbs passes for prune-regraft sampler per iteration.
    pub num_samples_prune_regraft: usize,
    /// Whether to update the concentration parameter each iteration.
    pub concentration_update: bool,
    /// Proposal distribution family.
    pub proposal: ProposalFamily,
    /// Probability of using the subtree Particle Gibbs update instead of the
    /// full-tree update.  0.0 = always full-tree (default).  Mirrors
    /// `subtree_update_prob` in PhyClone's `_run_main_sampler`.
    pub subtree_update_prob: f64,
}

impl Default for PhyCloneMcmcConfig {
    fn default() -> Self {
        Self {
            burnin: 100,
            num_iters: 1000,
            max_time: f64::INFINITY,
            print_freq: 100,
            thin: 1,
            num_particles: 100,
            resample_threshold: 0.5,
            num_samples_data_point: 1,
            num_samples_prune_regraft: 1,
            concentration_update: true,
            proposal: ProposalFamily::SemiAdapted,
            subtree_update_prob: 0.0,
        }
    }
}

/// PhyClone-compatible burn-in: run UnconditionalSMC + DataPoint + PruneRegraph for
/// `burnin` iterations; return the tree with the highest `log_p_one` seen.
///
/// Mirrors `_run_burnin` in `phyclone/run.py`.
pub fn run_phyclone_burnin(
    joint: &CompatTreeJointDistribution,
    data_points: &[CompatDataPoint],
    initial_tree: CompatTree,
    config: &PhyCloneMcmcConfig,
    chain_num: i32,
    chain_started: &std::time::Instant,
    rng: &mut impl Rng,
) -> Result<CompatTree, String> {
    use super::samplers::{CompatDataPointSampler, CompatPruneRegraphSampler};

    if config.burnin == 0 {
        return Ok(initial_tree);
    }

    let outlier_points = CompatTree::outlier_points(data_points);
    let dp_sampler = CompatDataPointSampler {
        joint: joint.clone(),
        outliers: joint.outlier_modelling_active,
    };
    let prg_sampler = CompatPruneRegraphSampler {
        joint: joint.clone(),
    };

    let mut tree = initial_tree;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_tree = tree.clone();

    for i in 0..config.burnin {
        if i % config.print_freq == 0 {
            eprintln!(
                "[phyclone-compat] burnin chain={} iter={} elapsed_s={:.3}",
                chain_num,
                i,
                chain_started.elapsed().as_secs_f64()
            );
        }

        // UnconditionalSMC pass.
        tree = run_unconditional_smc(
            joint,
            data_points,
            Some(&tree),
            config.num_particles,
            config.resample_threshold,
            config.proposal,
            rng,
        )?;

        // DataPoint Gibbs passes.
        for _ in 0..config.num_samples_data_point {
            tree = dp_sampler.sample_tree(data_points, &tree, rng)?;
        }

        // PruneRegraph passes.
        for _ in 0..config.num_samples_prune_regraft {
            tree = prg_sampler.sample_tree(data_points, &tree, rng)?;
        }

        // Track best tree.
        let score = joint.log_p_one_tree(&tree, &outlier_points)?;
        if score > best_score {
            best_score = score;
            best_tree = tree.clone();
        }

        if chain_started.elapsed().as_secs_f64() > config.max_time {
            break;
        }
    }

    Ok(best_tree)
}

/// PhyClone-compatible MCMC main loop.
///
/// Mirrors `_run_main_sampler` in `phyclone/run.py`:
/// - ParticleGibbs tree sampler
/// - `num_samples_data_point` × DataPointSampler
/// - `num_samples_prune_regraft` × PruneRegraphSampler
/// - Concentration update (optional)
/// - Thinning
pub fn run_phyclone_mcmc(
    joint: &mut CompatTreeJointDistribution,
    data_points: &[CompatDataPoint],
    initial_tree: CompatTree,
    config: &PhyCloneMcmcConfig,
    chain_num: i32,
    chain_started: &std::time::Instant,
    rng: &mut impl Rng,
) -> Result<Vec<CompatMcmcSample>, String> {
    use super::samplers::{
        CompatConcentrationSampler, CompatDataPointSampler, CompatPruneRegraphSampler,
    };

    if data_points.is_empty() {
        return Err("data_points must not be empty".to_string());
    }
    if config.thin == 0 {
        return Err("thin must be >= 1".to_string());
    }

    let mut dp_sampler = CompatDataPointSampler {
        joint: joint.clone(),
        outliers: joint.outlier_modelling_active,
    };
    let mut prg_sampler = CompatPruneRegraphSampler {
        joint: joint.clone(),
    };
    let conc_sampler = CompatConcentrationSampler::default();
    let mut pg_sampler = CompatParticleGibbsSampler {
        joint: joint.clone(),
        num_particles: config.num_particles,
        resample_threshold: config.resample_threshold,
        proposal: config.proposal,
    };
    let mut subtree_sampler = CompatParticleGibbsSubtreeSampler {
        joint: joint.clone(),
        num_particles: config.num_particles,
        resample_threshold: config.resample_threshold,
        proposal: config.proposal,
    };

    let mut tree = initial_tree;
    let capacity = config.num_iters / config.thin;
    let mut samples = Vec::with_capacity(capacity);
    let outlier_points = CompatTree::outlier_points(data_points);

    for i in 0..config.num_iters {
        if i % config.print_freq == 0 {
            eprintln!(
                "[phyclone-compat] main chain={} iter={} elapsed_s={:.3}",
                chain_num,
                i,
                chain_started.elapsed().as_secs_f64()
            );
        }

        // Particle Gibbs tree update: choose between subtree PG and full-tree PG.
        let use_subtree =
            config.subtree_update_prob > 0.0 && rng.random::<f64>() < config.subtree_update_prob;

        let pg_result = if use_subtree {
            subtree_sampler.sample_tree(data_points, &tree, rng)
        } else {
            pg_sampler.sample_tree(data_points, &tree, rng)
        };

        match pg_result {
            Ok(proposed) => tree = proposed,
            Err(err) => match SmcError::from_message(&err) {
                SmcError::DegenerateWeights => {
                    // Numerical degeneration can be treated as a reject move.
                }
                SmcError::NoConstrainedProposalMatched | SmcError::InvalidConstrainedPath => {
                    return Err(format!(
                        "particle gibbs constrained path mismatch at iter {}: {}",
                        i, err
                    ));
                }
                SmcError::Other(_) => {
                    return Err(format!(
                        "particle gibbs update failed at iter {}: {}",
                        i, err
                    ));
                }
            },
        }

        // DataPoint Gibbs passes.
        for _ in 0..config.num_samples_data_point {
            tree = dp_sampler.sample_tree(data_points, &tree, rng)?;
        }

        // PruneRegraph passes.
        for _ in 0..config.num_samples_prune_regraft {
            tree = prg_sampler.sample_tree(data_points, &tree, rng)?;
        }

        // Record sample.
        if i % config.thin == 0 {
            let (log_p, log_p_one) =
                joint.compute_log_p_and_log_p_one_tree(&tree, &outlier_points)?;
            samples.push(CompatMcmcSample {
                iter: i,
                tree: tree.clone(),
                alpha: joint.prior.alpha,
                log_p,
                log_p_one,
            });
        }

        if chain_started.elapsed().as_secs_f64() >= config.max_time {
            break;
        }

        // Concentration update (at end of iteration, matching PhyClone order).
        if config.concentration_update {
            let outlier_node_name = "-1";
            let (num_clusters, num_data_points) = {
                let mut nc = 0usize;
                let mut nd = 0usize;
                for node_id in tree.non_root_node_ids() {
                    if node_id == outlier_node_name {
                        continue;
                    }
                    let count = tree.node(node_id).map_or(0, |n| n.data_point_ids.len());
                    nc += 1;
                    nd += count;
                }
                (nc, nd)
            };
            let new_alpha =
                conc_sampler.sample(joint.prior.alpha, num_clusters, num_data_points, rng);
            joint.prior.alpha = new_alpha;
            dp_sampler.joint.prior.alpha = new_alpha;
            prg_sampler.joint.prior.alpha = new_alpha;
            pg_sampler.joint.prior.alpha = new_alpha;
            subtree_sampler.joint.prior.alpha = new_alpha;
        }
    }

    Ok(samples)
}

/// Run a single-pass unconditional SMC filter over `data_points`,
/// returning a sampled tree from the final particle swarm.
///
/// `current_tree`: the tree used to sample the data order via
/// `RootPermutationDistribution`. When `None`, the original `data_points` order is used
/// (e.g. the very first iteration before any tree exists).
///
/// `num_particles` controls the swarm size.
/// `resample_threshold` (0..1): resample if relative ESS falls below this value.
/// Uses the specified `proposal` distribution (default: `SemiAdapted`, matching PhyClone).
pub fn run_unconditional_smc(
    joint: &CompatTreeJointDistribution,
    data_points: &[CompatDataPoint],
    current_tree: Option<&CompatTree>,
    num_particles: usize,
    resample_threshold: f64,
    proposal: ProposalFamily,
    rng: &mut impl Rng,
) -> Result<CompatTree, String> {
    if data_points.is_empty() {
        return Err("data_points must not be empty".to_string());
    }
    if num_particles == 0 {
        return Err("num_particles must be > 0".to_string());
    }

    // Sample data order from the current tree topology, mirroring PhyClone's
    // UnconditionalSMCSampler which calls RootPermutationDistribution.sample.
    let ordered: Vec<CompatDataPoint> = if let Some(tree) = current_tree {
        let data_sigma = RootPermutationDistribution::sample(tree, rng);
        reorder_data_points(data_points, &data_sigma)?
    } else {
        data_points.to_vec()
    };
    let data_points = &ordered;
    let outlier_points = CompatTree::outlier_points(data_points);

    // Scorer closure: captures joint and outlier_points by reference.
    let scorer = |tree: &CompatTree| -> f64 {
        joint
            .log_p_tree(tree, &outlier_points)
            .unwrap_or(f64::NEG_INFINITY)
    };

    let adder = CompatTreeShellNodeAdder::default();

    // Determine grid dimensions from first data point.
    let first_dp = &data_points[0];
    let num_samples = first_dp.value.len();
    let num_grid_points = first_dp.value.first().map(|row| row.len()).unwrap_or(0);

    // Build a baseline empty tree compatible with these data points.
    // log_prior = -log(grid_size) matches PhyClone's TreeNode.__init__ which sets
    // log_p = np.full(grid_size, -np.log(grid_size[1])) for a uniform prior over the grid.
    let log_prior = if num_grid_points > 0 {
        -(num_grid_points as f64).ln()
    } else {
        0.0
    };
    let base_tree = CompatTree::new("root", num_samples, num_grid_points, log_prior, 8);

    // Initialise swarm: num_particles empty trees with uniform log weight.
    let uniform_log_weight = -(num_particles as f64).ln();
    let mut swarm = CompatParticleSwarm::new(
        (0..num_particles)
            .map(|_| CompatParticle::new(base_tree.clone(), uniform_log_weight))
            .collect(),
    );

    let num_dps = data_points.len();

    for (step, dp) in data_points.iter().enumerate() {
        let is_last = step == num_dps - 1;

        // Match PhyClone timing: resample before each update after step 0,
        // based on previous-step weights.
        if step > 0 {
            let rel_ess = swarm.relative_ess()?;
            if rel_ess <= resample_threshold {
                swarm.resample(rng)?;
            }
        }

        let dp_id = dp.idx;
        let dp_value = &dp.value;

        let mut new_particles = Vec::with_capacity(num_particles);

        for parent in &swarm.particles {
            let parent_log_p = scorer(&parent.holder.tree);
            let parent_log_weight = parent.log_weight;

            let proposal_set = build_proposal_set_for_kind(
                proposal,
                &parent.holder.tree,
                &adder,
                dp_id,
                dp_value,
                joint.outlier_modelling_active,
                step == 0,
                scorer,
            )?;

            if proposal_set.is_empty() {
                return Err(format!(
                    "proposal set is empty at step {} for dp_id={}",
                    step, dp_id
                ));
            }

            let proposal_idx = sample_discrete_from_log_weights(&proposal_set.log_q, rng)?;
            let log_q = proposal_set.log_q[proposal_idx];
            let new_tree = proposal_set.candidates[proposal_idx].tree.clone();

            // Compute log_p (and optionally log_p_one) for the new tree.
            let (new_log_p, new_log_p_one) =
                joint.compute_log_p_and_log_p_one_tree(&new_tree, &outlier_points)?;

            // Weight update: delta = log_target - log_p(parent) - log_q
            // At the last step the target is log_p_one (normalised probability).
            let log_target = if is_last { new_log_p_one } else { new_log_p };
            let delta = log_target - parent_log_p - log_q;

            new_particles.push(CompatParticle::new_with_log_p(
                new_tree,
                parent_log_weight + delta,
                new_log_p,
                new_log_p_one,
            ));
        }

        swarm.particles = new_particles;
    }

    // Normalise and sample.
    swarm.normalize_log_weights()?;
    let sampled_idx = sample_discrete_from_log_weights(
        &swarm
            .particles
            .iter()
            .map(|p| p.log_weight)
            .collect::<Vec<_>>(),
        rng,
    )?;

    Ok(swarm.particles[sampled_idx].holder.tree.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};
    use crate::phyclone::compat::distributions::CompatTreeJointDistribution;
    use crate::phyclone::compat::proposal::{
        build_bootstrap_proposal_set, CompatProposalKind as ProposalNodeKind,
        CompatTreeShellNodeAdder,
    };
    use crate::phyclone::compat::tree_model::CompatTree;

    fn approx_eq(lhs: f64, rhs: f64) {
        let delta = (lhs - rhs).abs();
        assert!(delta <= 1e-10, "lhs={} rhs={} delta={}", lhs, rhs, delta);
    }

    fn make_tree() -> CompatTree {
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "r0")
            .expect("add r0 should succeed");
        tree.add_child_node("root", "r1")
            .expect("add r1 should succeed");
        tree
    }

    fn assert_all_data_points_assigned_once(tree: &CompatTree, data_points: &[CompatDataPoint]) {
        let mut seen = std::collections::HashSet::new();

        for node_id in tree.non_root_node_ids() {
            if node_id == "-1" {
                continue;
            }

            if let Some(node) = tree.node(node_id) {
                for dp_id in &node.data_point_ids {
                    assert!(seen.insert(*dp_id), "duplicate dp_id {}", dp_id);
                }
            }
        }

        for dp_id in &tree.assigned_outliers {
            assert!(seen.insert(*dp_id), "duplicate outlier dp_id {}", dp_id);
        }

        for dp in data_points {
            assert!(seen.contains(&dp.idx), "missing dp_id {}", dp.idx);
        }

        assert_eq!(seen.len(), data_points.len());
    }

    fn make_subtree_test_data_points() -> Vec<CompatDataPoint> {
        (0..4)
            .map(|i| CompatDataPoint {
                idx: i,
                name: CompatDataPointName::Int(i as i64),
                mutation_ids: vec![format!("m{}", i)],
                sample_ids: vec!["s1".to_string()],
                value: vec![vec![-1.0 - 0.1 * i as f64, -0.6 - 0.1 * i as f64, -0.3]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect()
    }

    fn make_subtree_pg_initial_tree(data_points: &[CompatDataPoint]) -> CompatTree {
        let log_prior = -(data_points[0].value[0].len() as f64).ln();
        let mut init_tree = CompatTree::new("root", 1, data_points[0].value[0].len(), log_prior, 8);

        // Build a non-trivial chain root -> a -> b -> c where only non-root descendants
        // carry data. This guarantees sample_subtree_root_child picks b or c, so the
        // subtree sampler path is exercised instead of falling back to full-tree PG.
        init_tree
            .add_child_node("root", "a")
            .expect("add a should succeed");
        init_tree
            .add_child_node("a", "b")
            .expect("add b should succeed");
        init_tree
            .add_child_node("b", "c")
            .expect("add c should succeed");

        init_tree
            .node_mut("b")
            .expect("b exists")
            .add_data_point(data_points[0].idx, &data_points[0].value)
            .expect("assign dp0 to b");
        init_tree
            .node_mut("b")
            .expect("b exists")
            .add_data_point(data_points[1].idx, &data_points[1].value)
            .expect("assign dp1 to b");
        init_tree
            .node_mut("c")
            .expect("c exists")
            .add_data_point(data_points[2].idx, &data_points[2].value)
            .expect("assign dp2 to c");
        init_tree
            .node_mut("c")
            .expect("c exists")
            .add_data_point(data_points[3].idx, &data_points[3].value)
            .expect("assign dp3 to c");
        init_tree
            .update_all_nodes_postorder()
            .expect("initial tree should be internally consistent");

        init_tree
    }

    fn make_subtree_pg_initial_tree_with_outlier(data_points: &[CompatDataPoint]) -> CompatTree {
        let mut init_tree = make_subtree_pg_initial_tree(data_points);

        init_tree
            .node_mut("c")
            .expect("c exists")
            .remove_data_point(data_points[3].idx, &data_points[3].value)
            .expect("remove dp3 from c");
        init_tree.assigned_outliers.insert(data_points[3].idx);
        init_tree
            .update_all_nodes_postorder()
            .expect("initial outlier tree should be internally consistent");

        init_tree
    }

    #[test]
    fn smc_error_classification_is_typed() {
        assert_eq!(
            SmcError::from_message("[smc:degenerate_weights] all particle log weights are -inf"),
            SmcError::DegenerateWeights
        );
        assert_eq!(
            SmcError::from_message("[smc:no_constrained_proposal_matched] dp_id=0 target_node=n0"),
            SmcError::NoConstrainedProposalMatched
        );
        assert_eq!(
            SmcError::from_message("[smc:invalid_constrained_path] dp_id=0 target_node=None"),
            SmcError::InvalidConstrainedPath
        );
    }

    #[test]
    fn particle_weight_update_matches_log_target_minus_log_q() {
        let tree = make_tree();
        let adder = CompatTreeShellNodeAdder {
            outlier_node_id: "-1".to_string(),
            new_node_prefix: "new".to_string(),
        };

        let set = build_bootstrap_proposal_set(&tree, &adder, 99, &[vec![0.0; 3]], true, false)
            .expect("bootstrap proposal set should build");

        let proposal_index = set
            .candidates
            .iter()
            .position(|c| matches!(c.kind, ProposalNodeKind::Outlier))
            .expect("outlier candidate should exist");

        let old_log_weight = -2.5;
        let proposal_log_target = -5.2;
        let mut particle = CompatParticle::new(tree, old_log_weight);

        particle
            .update_from_proposal(&set, proposal_index, proposal_log_target)
            .expect("particle update should succeed");

        let expected = old_log_weight + proposal_log_target - set.log_q[proposal_index];
        approx_eq(particle.log_weight, expected);
        // After the outlier fix, outlier data points are tracked in assigned_outliers,
        // not as a graph node named "-1".
        assert!(particle.holder.tree.assigned_outliers.contains(&99));
    }

    #[test]
    fn bootstrap_proposal_uses_first_particle_semantics_at_initial_step() {
        let empty_tree = CompatTree::new("root", 1, 3, -1.0, 8);
        let adder = CompatTreeShellNodeAdder::default();
        let scorer = |_tree: &CompatTree| 0.0;

        let initial = build_proposal_set_for_kind(
            ProposalFamily::Bootstrap,
            &empty_tree,
            &adder,
            0,
            &[vec![0.0; 3]],
            true,
            true,
            scorer,
        )
        .expect("initial-step proposal should build");

        let non_initial = build_proposal_set_for_kind(
            ProposalFamily::Bootstrap,
            &empty_tree,
            &adder,
            0,
            &[vec![0.0; 3]],
            true,
            false,
            scorer,
        )
        .expect("non-initial-step proposal should build");

        let find_log_q = |set: &CompatProposalSet, is_outlier: bool| -> f64 {
            let idx = set
                .candidates
                .iter()
                .position(|c| match c.kind {
                    ProposalNodeKind::Outlier => is_outlier,
                    ProposalNodeKind::NewNode { ref children } => {
                        !is_outlier && children.is_empty()
                    }
                    ProposalNodeKind::ExistingNode { .. } => false,
                })
                .expect("target candidate should exist");
            set.log_q[idx]
        };

        // Bootstrap PhyClone semantics:
        // - initial: P(new)=0.9, P(outlier)=0.1
        // - non-initial with 0 roots: P(new)=0.45, P(outlier)=0.1
        approx_eq(find_log_q(&initial, false), (0.9_f64).ln());
        approx_eq(find_log_q(&initial, true), (0.1_f64).ln());
        approx_eq(find_log_q(&non_initial, false), (0.45_f64).ln());
        approx_eq(find_log_q(&non_initial, true), (0.1_f64).ln());
    }

    #[test]
    fn normalize_log_weights_and_ess_match_manual_formula() {
        let tree = make_tree();
        let mut swarm = CompatParticleSwarm::new(vec![
            CompatParticle::new(tree.clone(), -1.0),
            CompatParticle::new(tree.clone(), -2.0),
            CompatParticle::new(tree, -3.0),
        ]);

        swarm
            .normalize_log_weights()
            .expect("normalization should succeed");

        let probs: Vec<f64> = swarm.particles.iter().map(|p| p.log_weight.exp()).collect();
        let prob_sum: f64 = probs.iter().sum();
        approx_eq(prob_sum, 1.0);

        let expected_ess = 1.0 / probs.iter().map(|p| p * p).sum::<f64>();
        let ess = swarm
            .effective_sample_size()
            .expect("ESS computation should succeed");

        approx_eq(ess, expected_ess);
        assert!(ess > 1.0);
        assert!(ess <= swarm.particles.len() as f64);
    }

    #[test]
    fn resample_preserves_particle_count_and_normalises_weights() {
        let tree = make_tree();
        let mut swarm = CompatParticleSwarm::new(vec![
            CompatParticle::new(tree.clone(), -1.0),
            CompatParticle::new(tree.clone(), -2.0),
            CompatParticle::new(tree, -3.0),
        ]);

        let mut rng = rand::rng();
        swarm.resample(&mut rng).expect("resample should succeed");

        assert_eq!(swarm.particles.len(), 3);
        let expected_w = -(3_f64.ln());
        for p in &swarm.particles {
            approx_eq(p.log_weight, expected_w);
        }
    }

    #[test]
    fn run_unconditional_smc_returns_tree_with_all_data_points() {
        use crate::phyclone::compat::data::CompatDataPoint;
        use crate::phyclone::compat::distributions::CompatTreeJointDistribution;
        use crate::phyclone::compat::smc::run_unconditional_smc;

        // Build 3 simple data points with 1 sample, 3 grid points each.
        let dps: Vec<CompatDataPoint> = (0..3)
            .map(|i| CompatDataPoint {
                idx: i,
                name: crate::phyclone::compat::data::CompatDataPointName::Int(i as i64),
                mutation_ids: vec![],
                sample_ids: vec!["s1".to_string()],
                value: vec![vec![-1.0, -0.5, -0.2]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect();

        let joint = CompatTreeJointDistribution::default();
        let mut rng = rand::rng();

        let tree = run_unconditional_smc(
            &joint,
            &dps,
            None,
            5,
            0.5,
            ProposalFamily::FullyAdapted,
            &mut rng,
        )
        .expect("unconditional SMC should succeed");

        // Every data point must appear in the tree exactly once.
        let all_ids: std::collections::HashSet<usize> = tree
            .nodes
            .values()
            .flat_map(|n| n.data_point_ids.iter().copied())
            .collect();

        for dp in &dps {
            assert!(
                all_ids.contains(&dp.idx),
                "dp idx {} not found in tree",
                dp.idx
            );
        }
    }

    #[test]
    fn constrained_particle_sequence_assigns_all_data_points() {
        use crate::phyclone::compat::distributions::CompatTreeJointDistribution;

        // Build a simple retained tree: root -> n0 (dp0), root -> n1 (dp1)
        let mut tree = CompatTree::new("root", 1, 3, -1.0, 8);
        tree.add_child_node("root", "n0")
            .expect("add n0 should succeed");
        tree.add_child_node("root", "n1")
            .expect("add n1 should succeed");

        let dp0 = vec![vec![-1.0, -0.5, -0.2]];
        let dp1 = vec![vec![-0.9, -0.4, -0.1]];

        tree.node_mut("n0")
            .expect("n0 must exist")
            .add_data_point(0, &dp0)
            .expect("add dp0 should succeed");
        tree.update_path_to_root("n0")
            .expect("update n0 path should succeed");

        tree.node_mut("n1")
            .expect("n1 must exist")
            .add_data_point(1, &dp1)
            .expect("add dp1 should succeed");
        tree.update_path_to_root("n1")
            .expect("update n1 path should succeed");

        let dps: Vec<CompatDataPoint> = (0..2)
            .map(|i| CompatDataPoint {
                idx: i,
                name: crate::phyclone::compat::data::CompatDataPointName::Int(i as i64),
                mutation_ids: vec![format!("m{}", i)],
                sample_ids: vec!["s0".to_string()],
                value: vec![vec![-1.0, -0.5, -0.2]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect();

        let joint = CompatTreeJointDistribution::default();
        let steps =
            build_constrained_particle_sequence(&tree, &dps, &joint, ProposalFamily::FullyAdapted)
                .expect("constrained particle sequence should build");

        assert_eq!(steps.len(), 2, "should have one step per data point");

        // Final tree must contain both data points.
        let final_tree = &steps.last().unwrap().tree;
        let all_ids: std::collections::HashSet<usize> = final_tree
            .nodes
            .values()
            .flat_map(|n| n.data_point_ids.iter().copied())
            .collect();
        assert!(all_ids.contains(&0), "dp 0 not in final constrained tree");
        assert!(all_ids.contains(&1), "dp 1 not in final constrained tree");
        // log_p and log_q must be finite.
        assert!(steps[0].log_p.is_finite());
        assert!(steps[0].log_q.is_finite());
        assert!(steps[1].log_p.is_finite());
        assert!(steps[1].log_q.is_finite());
    }

    #[test]
    fn run_conditional_smc_returns_tree_with_all_data_points() {
        let dps: Vec<CompatDataPoint> = (0..3)
            .map(|i| CompatDataPoint {
                idx: i,
                name: CompatDataPointName::Int(i as i64),
                mutation_ids: vec![format!("m{}", i)],
                sample_ids: vec!["s1".to_string()],
                value: vec![vec![-1.0, -0.5, -0.2]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect();

        let joint = CompatTreeJointDistribution::default();
        let mut rng = rand::rng();

        let retained = run_unconditional_smc(
            &joint,
            &dps,
            None,
            6,
            0.5,
            ProposalFamily::FullyAdapted,
            &mut rng,
        )
        .expect("unconditional SMC should succeed");

        let tree = run_conditional_smc(
            &joint,
            &dps,
            &retained,
            6,
            0.5,
            ProposalFamily::FullyAdapted,
            &mut rng,
        )
        .expect("conditional SMC should succeed");

        assert_all_data_points_assigned_once(&tree, &dps);
    }

    #[test]
    fn subtree_pg_sample_tree_returns_valid_tree() {
        let dps = make_subtree_test_data_points();
        let current_tree = make_subtree_pg_initial_tree(&dps);
        let joint = CompatTreeJointDistribution::default();
        let sampler = CompatParticleGibbsSubtreeSampler {
            joint: joint.clone(),
            num_particles: 10,
            resample_threshold: 0.5,
            proposal: ProposalFamily::SemiAdapted,
        };
        let mut rng = rand::rng();

        let tree = sampler
            .sample_tree(&dps, &current_tree, &mut rng)
            .expect("subtree PG should return a valid tree");

        let (log_p, log_p_one) = joint
            .compute_log_p_and_log_p_one_tree(&tree, &CompatTree::outlier_points(&dps))
            .expect("score computation should succeed");

        assert!(log_p.is_finite());
        assert!(log_p_one.is_finite());
        assert_all_data_points_assigned_once(&tree, &dps);
    }

    #[test]
    fn subtree_pg_sample_tree_preserves_outlier_assignment_exactly_once() {
        let dps = make_subtree_test_data_points();
        let current_tree = make_subtree_pg_initial_tree_with_outlier(&dps);
        let joint = CompatTreeJointDistribution {
            prior: Default::default(),
            outlier_modelling_active: true,
        };
        let sampler = CompatParticleGibbsSubtreeSampler {
            joint: joint.clone(),
            num_particles: 10,
            resample_threshold: 0.5,
            proposal: ProposalFamily::SemiAdapted,
        };
        let mut rng = rand::rng();

        let tree = sampler
            .sample_tree(&dps, &current_tree, &mut rng)
            .expect("subtree PG with outliers should return a valid tree");

        let (log_p, log_p_one) = joint
            .compute_log_p_and_log_p_one_tree(&tree, &CompatTree::outlier_points(&dps))
            .expect("score computation should succeed");

        assert!(log_p.is_finite());
        assert!(log_p_one.is_finite());
        assert_all_data_points_assigned_once(&tree, &dps);

        let outlier_dp_id = dps[3].idx;
        let outlier_count = usize::from(tree.assigned_outliers.contains(&outlier_dp_id));
        let node_count = tree
            .non_root_node_ids()
            .filter_map(|node_id| tree.node(node_id))
            .filter(|node| node.data_point_ids.contains(&outlier_dp_id))
            .count();
        assert_eq!(
            outlier_count + node_count,
            1,
            "outlier dp_id {} must appear exactly once in either assigned_outliers or an ordinary node",
            outlier_dp_id
        );
    }

    #[test]
    fn run_compat_mcmc_inner_returns_expected_count() {
        use crate::phyclone::compat::data::CompatDataPoint;
        use crate::phyclone::compat::distributions::CompatTreeJointDistribution;

        let dps: Vec<CompatDataPoint> = (0..2)
            .map(|i| CompatDataPoint {
                idx: i,
                name: crate::phyclone::compat::data::CompatDataPointName::Int(i as i64),
                mutation_ids: vec![format!("m{}", i)],
                sample_ids: vec!["s1".to_string()],
                value: vec![vec![-1.0, -0.5, -0.2]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect();

        let joint = CompatTreeJointDistribution::default();
        let mut rng = rand::rng();
        let init_tree = run_unconditional_smc(
            &joint,
            &dps,
            None,
            6,
            0.5,
            ProposalFamily::FullyAdapted,
            &mut rng,
        )
        .expect("unconditional init should succeed");

        let sampler = CompatParticleGibbsSampler {
            joint,
            num_particles: 6,
            resample_threshold: 0.5,
            proposal: ProposalFamily::FullyAdapted,
        };

        let samples = run_compat_mcmc_inner(&sampler, &dps, init_tree, 2, 4, 1, &mut rng)
            .expect("mcmc inner should succeed");
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn run_phyclone_mcmc_iter_numbers_match_thin() {
        use crate::phyclone::compat::data::CompatDataPoint;
        use crate::phyclone::compat::distributions::CompatTreeJointDistribution;

        let dps: Vec<CompatDataPoint> = (0..2)
            .map(|i| CompatDataPoint {
                idx: i,
                name: crate::phyclone::compat::data::CompatDataPointName::Int(i as i64),
                mutation_ids: vec![format!("m{}", i)],
                sample_ids: vec!["s1".to_string()],
                value: vec![vec![-1.0, -0.5, -0.2]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect();

        let mut joint = CompatTreeJointDistribution::default();
        let mut rng = rand::rng();
        let init_tree = run_unconditional_smc(
            &joint,
            &dps,
            None,
            4,
            0.5,
            ProposalFamily::FullyAdapted,
            &mut rng,
        )
        .expect("initial tree");

        let config = PhyCloneMcmcConfig {
            burnin: 0,
            num_iters: 6,
            max_time: f64::INFINITY,
            print_freq: 100,
            thin: 2,
            num_particles: 4,
            resample_threshold: 0.5,
            proposal: ProposalFamily::FullyAdapted,
            num_samples_data_point: 1,
            num_samples_prune_regraft: 1,
            concentration_update: false,
            subtree_update_prob: 0.0,
        };

        let samples = run_phyclone_mcmc(
            &mut joint,
            &dps,
            init_tree,
            &config,
            0,
            &std::time::Instant::now(),
            &mut rng,
        )
        .expect("mcmc should succeed");

        // num_iters=6, thin=2 → 3 samples at iter 0, 2, 4
        assert_eq!(samples.len(), 3);
        let iters: Vec<usize> = samples.iter().map(|sample| sample.iter).collect();
        assert_eq!(iters, vec![0, 2, 4]);
    }

    #[test]
    fn run_phyclone_mcmc_with_subtree_prob_one_produces_samples() {
        let dps = make_subtree_test_data_points();
        let mut joint = CompatTreeJointDistribution::default();
        let mut rng = rand::rng();
        let init_tree = make_subtree_pg_initial_tree(&dps);

        subtree_pg_reset_core_path_hits();

        let config = PhyCloneMcmcConfig {
            burnin: 0,
            num_iters: 5,
            max_time: f64::INFINITY,
            print_freq: 100,
            thin: 1,
            num_particles: 10,
            resample_threshold: 0.5,
            proposal: ProposalFamily::SemiAdapted,
            num_samples_data_point: 1,
            num_samples_prune_regraft: 1,
            concentration_update: false,
            subtree_update_prob: 1.0,
        };

        let samples = run_phyclone_mcmc(
            &mut joint,
            &dps,
            init_tree,
            &config,
            0,
            &std::time::Instant::now(),
            &mut rng,
        )
        .expect("mcmc should succeed with subtree_update_prob=1.0");

        assert!(!samples.is_empty());
        assert!(
            subtree_pg_core_path_hits() > 0,
            "expected at least one non-fallback subtree PG update"
        );

        for sample in &samples {
            assert_all_data_points_assigned_once(&sample.tree, &dps);
        }
    }
}
