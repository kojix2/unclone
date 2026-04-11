use serde::{Deserialize, Serialize};

use super::data::CompatDataPoint;
use super::distributions::CompatTreeJointDistribution;
use super::smc::CompatMcmcSample;
use super::smc::ProposalFamily;
use super::tree_model::CompatTree;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatSampleProfile {
    pub sample_id: String,
    pub ref_counts: i32,
    pub alt_counts: i32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatOutlierSampleObservation {
    pub sample_id: String,
    pub ref_counts: i32,
    pub alt_counts: i32,
    pub major_cn: i32,
    pub minor_cn: i32,
    pub normal_cn: i32,
    pub tumour_content: f64,
    pub error_rate: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatOutlierDataPoint {
    pub name: String,
    pub outlier_prob: f64,
    pub outlier_prob_not: f64,
    pub outlier_marginal_prob: f64,
    #[serde(default)]
    pub loss_log_prob: f64,
    #[serde(default)]
    pub sample_observations: Vec<CompatOutlierSampleObservation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatClusterSummary {
    pub cluster_id: i32,
    pub mutation_ids: Vec<String>,
    pub sample_ids: Vec<String>,
    #[serde(default)]
    pub sample_profiles: Vec<CompatSampleProfile>,
    #[serde(default)]
    pub outlier_data_points: Vec<CompatOutlierDataPoint>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatOutlierAssignment {
    pub mutation_id: String,
    pub is_outlier: bool,
    /// Log-odds of outlier vs in-tree assignment.
    /// `None` when not computed (tyclone does not currently compute this value).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub log_odds_outlier_vs_in_tree: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatTraceMetadata {
    pub num_chains: i32,
    pub num_iters: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatTraceNode {
    pub id: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cluster_ids: Vec<i32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatTraceEdge {
    pub parent: String,
    pub child: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatTraceTree {
    pub nodes: Vec<CompatTraceNode>,
    pub edges: Vec<CompatTraceEdge>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatTraceRecord {
    pub schema_version: i32,
    pub chain: i32,
    pub iter: i32,
    pub log_p: f64,
    pub log_p_one: f64,
    pub topology_id: String,
    pub tree: CompatTraceTree,
    pub clusters: Vec<CompatClusterSummary>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outlier_assignments: Vec<CompatOutlierAssignment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<CompatTraceMetadata>,
}

pub fn compat_trace_records_to_jsonl(
    records: &[CompatTraceRecord],
) -> Result<String, serde_json::Error> {
    let mut output = String::new();
    for record in records {
        output.push_str(&serde_json::to_string(record)?);
        output.push('\n');
    }
    Ok(output)
}

/// Convert a `CompatTree` produced by `run_unconditional_smc` into a `CompatTraceRecord`.
///
/// `data_points` and `clusters` are used to build human-readable mutation-id / cluster lists.
/// `chain` / `iter` / `seed` are forwarded as metadata.
#[allow(clippy::too_many_arguments)]
fn tree_to_trace_record(
    tree: &CompatTree,
    joint: &CompatTreeJointDistribution,
    data_points: &[CompatDataPoint],
    clusters: &[CompatClusterSummary],
    chain: i32,
    iter: i32,
    seed: Option<u64>,
    num_chains: i32,
    num_iters: i32,
) -> CompatTraceRecord {
    use crate::phyclone::{TRACE_CLUSTER_KIND, TRACE_ROOT_ID, TRACE_ROOT_KIND};

    // Build outlier_points from data_points for score computation.
    let outlier_points = CompatTree::outlier_points(data_points);
    let (log_p, log_p_one) = joint
        .compute_log_p_and_log_p_one_tree(tree, &outlier_points)
        .unwrap_or((-1.0e9, -1.0e9));

    // Build a mapping from DataPointId -> cluster_id using the cluster summaries and
    // the data_point idx (loader guarantees cluster_id == sorted cluster index).
    let dp_idx_to_cluster_id: std::collections::HashMap<usize, i32> = data_points
        .iter()
        .enumerate()
        .filter_map(|(seq, dp)| clusters.get(seq).map(|c| (dp.idx, c.cluster_id)))
        .collect();

    // Non-root node ids (excluding the virtual outlier node "-1").
    let outlier_node_id = "-1";
    let mut non_root_non_outlier_ids: Vec<&str> = tree
        .nodes
        .keys()
        .filter(|id| id.as_str() != tree.root_node_id.as_str() && id.as_str() != outlier_node_id)
        .map(|id| id.as_str())
        .collect();
    non_root_non_outlier_ids.sort();

    // Build trace nodes.
    let mut nodes = Vec::with_capacity(non_root_non_outlier_ids.len() + 1);
    nodes.push(CompatTraceNode {
        id: TRACE_ROOT_ID.to_string(),
        kind: TRACE_ROOT_KIND.to_string(),
        cluster_ids: Vec::new(),
    });
    for node_id in &non_root_non_outlier_ids {
        // Collect all cluster_ids of data points in this node (sorted, deduped).
        let mut cluster_ids: Vec<i32> = tree
            .nodes
            .get(*node_id)
            .map(|n| {
                n.data_point_ids
                    .iter()
                    .filter_map(|dp_id| dp_idx_to_cluster_id.get(dp_id).copied())
                    .collect()
            })
            .unwrap_or_default();
        cluster_ids.sort();
        cluster_ids.dedup();
        nodes.push(CompatTraceNode {
            id: node_id.to_string(),
            kind: TRACE_CLUSTER_KIND.to_string(),
            cluster_ids,
        });
    }

    // Build topology_id and edges.
    // topology_id encodes the full tree structure (edges) using sorted cluster data-point sets
    // so that it is stable across runs and independent of internal node-id assignment.
    let mut edges: Vec<CompatTraceEdge> = Vec::new();
    for node_id in &non_root_non_outlier_ids {
        let parent = tree
            .parent_of(node_id)
            .ok()
            .flatten()
            .filter(|p| *p != tree.root_node_id.as_str())
            .unwrap_or(TRACE_ROOT_ID);
        edges.push(CompatTraceEdge {
            parent: parent.to_string(),
            child: node_id.to_string(),
        });
    }
    // Build a canonical topology_id from edges expressed as descendant-clade cluster-id sets.
    // Each node is keyed by the sorted union of cluster_ids in its entire subtree, matching
    // the clade representation used by Crystal-side clade-support computation.  Using the full
    // subtree (rather than direct cluster_ids only) makes the key stable even when the same
    // biological topology is sampled via different internal node-id assignments.
    let mut children_map_smc: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    for edge in &edges {
        children_map_smc
            .entry(edge.parent.as_str())
            .or_default()
            .push(edge.child.as_str());
    }
    let subtree_cluster_key_smc = |root: &str| -> String {
        if root == TRACE_ROOT_ID {
            return "root".to_string();
        }
        let mut stack = vec![root];
        let mut all_cids: Vec<i32> = Vec::new();
        while let Some(nid) = stack.pop() {
            if let Some(n) = tree.nodes.get(nid) {
                for dp_id in &n.data_point_ids {
                    if let Some(&cid) = dp_idx_to_cluster_id.get(dp_id) {
                        all_cids.push(cid);
                    }
                }
            }
            if let Some(children) = children_map_smc.get(nid) {
                for &child in children {
                    stack.push(child);
                }
            }
        }
        all_cids.sort();
        all_cids.dedup();
        format!(
            "c{}",
            all_cids
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join("_")
        )
    };
    let mut edge_keys: Vec<String> = edges
        .iter()
        .map(|e| {
            format!(
                "{}>{}",
                subtree_cluster_key_smc(&e.parent),
                subtree_cluster_key_smc(&e.child)
            )
        })
        .collect();
    edge_keys.sort();
    let topology_id = format!("smc-{}", edge_keys.join("|"));

    // Build outlier assignments.
    // log_odds_outlier_vs_in_tree is not computed by tyclone; emit None to avoid misleading 0.0.
    let outlier_assignments: Vec<CompatOutlierAssignment> = data_points
        .iter()
        .flat_map(|dp| dp.mutation_ids.iter().map(move |m| (dp.idx, m)))
        .map(|(dp_idx, mutation_id)| CompatOutlierAssignment {
            mutation_id: mutation_id.clone(),
            is_outlier: tree.assigned_outliers.contains(&dp_idx),
            log_odds_outlier_vs_in_tree: None,
        })
        .collect();

    CompatTraceRecord {
        schema_version: crate::phyclone::TRACE_SCHEMA_VERSION,
        chain,
        iter,
        log_p,
        log_p_one,
        topology_id,
        tree: CompatTraceTree { nodes, edges },
        clusters: clusters.to_vec(),
        outlier_assignments,
        metadata: Some(CompatTraceMetadata {
            num_chains,
            num_iters,
            seed,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn mcmc_sample_to_trace_record(
    sample: &CompatMcmcSample,
    data_points: &[CompatDataPoint],
    clusters: &[CompatClusterSummary],
    chain: i32,
    seed: Option<u64>,
    num_chains: i32,
    num_iters: i32,
) -> CompatTraceRecord {
    use crate::phyclone::{TRACE_CLUSTER_KIND, TRACE_ROOT_ID, TRACE_ROOT_KIND};

    let tree = &sample.tree;

    let dp_idx_to_cluster_id: std::collections::HashMap<usize, i32> = data_points
        .iter()
        .enumerate()
        .filter_map(|(seq, dp)| clusters.get(seq).map(|c| (dp.idx, c.cluster_id)))
        .collect();

    let outlier_node_id = "-1";
    let mut non_root_non_outlier_ids: Vec<&str> = tree
        .nodes
        .keys()
        .filter(|id| id.as_str() != tree.root_node_id.as_str() && id.as_str() != outlier_node_id)
        .map(|id| id.as_str())
        .collect();
    non_root_non_outlier_ids.sort();

    let mut nodes = Vec::with_capacity(non_root_non_outlier_ids.len() + 1);
    nodes.push(CompatTraceNode {
        id: TRACE_ROOT_ID.to_string(),
        kind: TRACE_ROOT_KIND.to_string(),
        cluster_ids: Vec::new(),
    });
    for node_id in &non_root_non_outlier_ids {
        let mut cluster_ids: Vec<i32> = tree
            .nodes
            .get(*node_id)
            .map(|n| {
                n.data_point_ids
                    .iter()
                    .filter_map(|dp_id| dp_idx_to_cluster_id.get(dp_id).copied())
                    .collect()
            })
            .unwrap_or_default();
        cluster_ids.sort();
        cluster_ids.dedup();
        nodes.push(CompatTraceNode {
            id: node_id.to_string(),
            kind: TRACE_CLUSTER_KIND.to_string(),
            cluster_ids,
        });
    }

    let mut edges: Vec<CompatTraceEdge> = Vec::new();
    for node_id in &non_root_non_outlier_ids {
        let parent = tree
            .parent_of(node_id)
            .ok()
            .flatten()
            .filter(|p| *p != tree.root_node_id.as_str())
            .unwrap_or(TRACE_ROOT_ID);
        edges.push(CompatTraceEdge {
            parent: parent.to_string(),
            child: node_id.to_string(),
        });
    }
    // Build a canonical topology_id from edges expressed as descendant-clade cluster-id sets.
    // Each node is keyed by the sorted union of cluster_ids in its entire subtree.
    let mut children_map_mcmc: std::collections::HashMap<&str, Vec<&str>> =
        std::collections::HashMap::new();
    for edge in &edges {
        children_map_mcmc
            .entry(edge.parent.as_str())
            .or_default()
            .push(edge.child.as_str());
    }
    let subtree_cluster_key_mcmc = |root: &str| -> String {
        if root == TRACE_ROOT_ID {
            return "root".to_string();
        }
        let mut stack = vec![root];
        let mut all_cids: Vec<i32> = Vec::new();
        while let Some(nid) = stack.pop() {
            if let Some(n) = tree.nodes.get(nid) {
                for dp_id in &n.data_point_ids {
                    if let Some(&cid) = dp_idx_to_cluster_id.get(dp_id) {
                        all_cids.push(cid);
                    }
                }
            }
            if let Some(children) = children_map_mcmc.get(nid) {
                for &child in children {
                    stack.push(child);
                }
            }
        }
        all_cids.sort();
        all_cids.dedup();
        format!(
            "c{}",
            all_cids
                .iter()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join("_")
        )
    };
    let mut edge_keys: Vec<String> = edges
        .iter()
        .map(|e| {
            format!(
                "{}>{}",
                subtree_cluster_key_mcmc(&e.parent),
                subtree_cluster_key_mcmc(&e.child)
            )
        })
        .collect();
    edge_keys.sort();
    let topology_id = format!("smc-{}", edge_keys.join("|"));

    // log_odds_outlier_vs_in_tree is not computed by tyclone; emit None to avoid misleading 0.0.
    let outlier_assignments: Vec<CompatOutlierAssignment> = data_points
        .iter()
        .flat_map(|dp| dp.mutation_ids.iter().map(move |m| (dp.idx, m)))
        .map(|(dp_idx, mutation_id)| CompatOutlierAssignment {
            mutation_id: mutation_id.clone(),
            is_outlier: tree.assigned_outliers.contains(&dp_idx),
            log_odds_outlier_vs_in_tree: None,
        })
        .collect();

    CompatTraceRecord {
        schema_version: crate::phyclone::TRACE_SCHEMA_VERSION,
        chain,
        iter: sample.iter as i32,
        log_p: sample.log_p,
        log_p_one: sample.log_p_one,
        topology_id,
        tree: CompatTraceTree { nodes, edges },
        clusters: clusters.to_vec(),
        outlier_assignments,
        metadata: Some(CompatTraceMetadata {
            num_chains,
            num_iters,
            seed,
        }),
    }
}

/// Run multi-chain Particle Gibbs (conditional SMC) and emit trace records.
///
/// Each chain is initialised with one unconditional SMC draw, then advanced with
/// conditional SMC under burn-in / thinning controls.
#[allow(clippy::too_many_arguments)]
pub fn run_compat_mcmc_traces_from_data_points(
    num_chains: i32,
    num_iters: i32,
    burnin: usize,
    thin: usize,
    data_points: &[CompatDataPoint],
    clusters: &[CompatClusterSummary],
    seed: Option<u64>,
    num_particles: usize,
    resample_threshold: f64,
) -> Vec<CompatTraceRecord> {
    use super::smc::{
        run_compat_mcmc_inner, run_unconditional_smc, CompatParticleGibbsSampler, ProposalFamily,
    };
    use rand::SeedableRng;

    if data_points.is_empty() || clusters.is_empty() || num_iters <= 0 || num_chains <= 0 {
        return Vec::new();
    }

    let joint = CompatTreeJointDistribution {
        outlier_modelling_active: data_points.iter().any(|dp| dp.raw_outlier_prob > 0.0),
        ..CompatTreeJointDistribution::default()
    };

    let mut records =
        Vec::with_capacity((num_chains.max(0) as usize) * (num_iters.max(0) as usize));

    for chain in 0..num_chains {
        let mut rng: rand::rngs::SmallRng = match seed {
            Some(s) => {
                rand::rngs::SmallRng::seed_from_u64(s ^ (chain as u64 * 6364136223846793005))
            }
            None => rand::rngs::SmallRng::seed_from_u64(rand::random::<u64>()),
        };

        let init_tree = match run_unconditional_smc(
            &joint,
            data_points,
            None,
            num_particles,
            resample_threshold,
            ProposalFamily::FullyAdapted,
            &mut rng,
        ) {
            Ok(t) => t,
            Err(_) => continue,
        };

        let sampler = CompatParticleGibbsSampler {
            joint: joint.clone(),
            num_particles,
            resample_threshold,
            proposal: ProposalFamily::FullyAdapted,
        };

        let samples = match run_compat_mcmc_inner(
            &sampler,
            data_points,
            init_tree,
            burnin,
            num_iters as usize,
            thin,
            &mut rng,
        ) {
            Ok(s) => s,
            Err(_) => continue,
        };

        for (iter, tree) in samples.iter().enumerate() {
            records.push(tree_to_trace_record(
                tree,
                &joint,
                data_points,
                clusters,
                chain,
                iter as i32,
                seed,
                num_chains,
                num_iters,
            ));
        }
    }

    records
}

/// Run multi-chain PhyClone-compatible burn-in + main MCMC and emit trace records.
#[allow(clippy::too_many_arguments)]
pub fn run_phyclone_mcmc_traces_from_data_points(
    num_chains: i32,
    num_iters: i32,
    burnin: usize,
    thin: usize,
    data_points: &[CompatDataPoint],
    clusters: &[CompatClusterSummary],
    seed: Option<u64>,
    num_particles: usize,
    resample_threshold: f64,
    max_time: f64,
    print_freq: usize,
    proposal: ProposalFamily,
    num_samples_data_point: usize,
    num_samples_prune_regraft: usize,
    concentration_update: bool,
    concentration_value: f64,
    subtree_update_prob: f64,
) -> Result<Vec<CompatTraceRecord>, String> {
    use super::smc::{run_phyclone_burnin, run_phyclone_mcmc, PhyCloneMcmcConfig};
    use super::tree_model::build_single_node_tree;
    use rand::SeedableRng;

    if data_points.is_empty() {
        return Err(
            "run_phyclone_mcmc_traces_from_data_points: data_points must not be empty".to_string(),
        );
    }
    if clusters.is_empty() {
        return Err(
            "run_phyclone_mcmc_traces_from_data_points: clusters must not be empty".to_string(),
        );
    }
    if num_iters <= 0 {
        return Err("run_phyclone_mcmc_traces_from_data_points: num_iters must be > 0".to_string());
    }
    if num_chains <= 0 {
        return Err(
            "run_phyclone_mcmc_traces_from_data_points: num_chains must be > 0".to_string(),
        );
    }

    let mut records =
        Vec::with_capacity((num_chains.max(0) as usize) * (num_iters.max(0) as usize));

    for chain in 0..num_chains {
        let mut rng: rand::rngs::SmallRng = match seed {
            Some(s) => {
                rand::rngs::SmallRng::seed_from_u64(s ^ (chain as u64 * 6364136223846793005))
            }
            None => rand::rngs::SmallRng::seed_from_u64(rand::random::<u64>()),
        };

        let mut joint = CompatTreeJointDistribution {
            prior: super::distributions::CompatFscrpDistribution {
                alpha: concentration_value,
                ..super::distributions::CompatFscrpDistribution::default()
            },
            outlier_modelling_active: data_points.iter().any(|dp| dp.raw_outlier_prob > 0.0),
        };

        let config = PhyCloneMcmcConfig {
            burnin,
            // PhyClone semantics: burn-in and main-iteration counts are separate.
            num_iters: num_iters as usize,
            max_time,
            print_freq,
            thin,
            num_particles,
            resample_threshold,
            num_samples_data_point,
            num_samples_prune_regraft,
            concentration_update,
            proposal,
            subtree_update_prob,
        };

        let num_samples = data_points[0].value.len();
        let grid_size = data_points[0]
            .value
            .first()
            .map(|row| row.len())
            .unwrap_or(0);
        if num_samples == 0 || grid_size == 0 {
            return Err(format!(
                "chain {}: invalid data shape for initial tree (num_samples={}, grid_size={})",
                chain, num_samples, grid_size
            ));
        }

        let log_prior = -(grid_size as f64).ln();
        let initial_tree =
            build_single_node_tree(data_points, num_samples, grid_size, log_prior)
                .map_err(|e| format!("chain {}: initial tree construction failed: {}", chain, e))?;
        let chain_started = std::time::Instant::now();

        let post_burnin_tree = run_phyclone_burnin(
            &joint,
            data_points,
            initial_tree,
            &config,
            chain,
            &chain_started,
            &mut rng,
        )
        .map_err(|e| format!("chain {}: burn-in failed: {}", chain, e))?;

        let samples = run_phyclone_mcmc(
            &mut joint,
            data_points,
            post_burnin_tree,
            &config,
            chain,
            &chain_started,
            &mut rng,
        )
        .map_err(|e| format!("chain {}: main MCMC failed: {}", chain, e))?;

        for sample in &samples {
            records.push(mcmc_sample_to_trace_record(
                sample,
                data_points,
                clusters,
                chain,
                seed,
                num_chains,
                num_iters,
            ));
        }
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phyclone::TRACE_SCHEMA_VERSION;

    #[test]
    fn run_compat_mcmc_traces_returns_correct_count() {
        use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};

        let clusters = vec![
            CompatClusterSummary {
                cluster_id: 0,
                mutation_ids: vec!["m0".to_string()],
                sample_ids: vec!["s0".to_string()],
                sample_profiles: Vec::new(),
                outlier_data_points: Vec::new(),
            },
            CompatClusterSummary {
                cluster_id: 1,
                mutation_ids: vec!["m1".to_string()],
                sample_ids: vec!["s0".to_string()],
                sample_profiles: Vec::new(),
                outlier_data_points: Vec::new(),
            },
        ];

        let data_points: Vec<CompatDataPoint> = (0..2)
            .map(|i| CompatDataPoint {
                idx: i,
                name: CompatDataPointName::Int(i as i64),
                mutation_ids: vec![format!("m{}", i)],
                sample_ids: vec!["s0".to_string()],
                value: vec![vec![-1.2, -0.8, -0.4]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect();

        let records = run_compat_mcmc_traces_from_data_points(
            2,
            3,
            0,
            1,
            &data_points,
            &clusters,
            Some(123),
            8,
            0.5,
        );

        assert_eq!(records.len(), 6, "expected 2 chains x 3 iters = 6 records");
        assert!(records
            .iter()
            .all(|r| r.schema_version == TRACE_SCHEMA_VERSION));
        assert!(records.iter().all(|r| r.log_p.is_finite()));
        assert!(records.iter().all(|r| r.log_p_one.is_finite()));
    }

    /// With thin=2, the output record count equals ceil(num_iters / thin).
    #[test]
    fn thin_reduces_output_count() {
        use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};

        let clusters = vec![
            CompatClusterSummary {
                cluster_id: 0,
                mutation_ids: vec!["m0".to_string()],
                sample_ids: vec!["s0".to_string()],
                sample_profiles: Vec::new(),
                outlier_data_points: Vec::new(),
            },
            CompatClusterSummary {
                cluster_id: 1,
                mutation_ids: vec!["m1".to_string()],
                sample_ids: vec!["s0".to_string()],
                sample_profiles: Vec::new(),
                outlier_data_points: Vec::new(),
            },
        ];
        let data_points: Vec<CompatDataPoint> = (0..2)
            .map(|i| CompatDataPoint {
                idx: i,
                name: CompatDataPointName::Int(i as i64),
                mutation_ids: vec![format!("m{}", i)],
                sample_ids: vec!["s0".to_string()],
                value: vec![vec![-1.2, -0.8, -0.4]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect();

        // num_iters=6, thin=2 -> 3 records per chain; 2 chains -> 6 total
        let thin2 = run_compat_mcmc_traces_from_data_points(
            2,
            6,
            0,
            2,
            &data_points,
            &clusters,
            Some(7),
            8,
            0.5,
        );
        // thin=1 -> 6 records per chain; 2 chains -> 12 total
        let thin1 = run_compat_mcmc_traces_from_data_points(
            2,
            6,
            0,
            1,
            &data_points,
            &clusters,
            Some(7),
            8,
            0.5,
        );
        assert_eq!(thin1.len(), 12, "thin=1: 2 chains x 6 iters");
        assert_eq!(thin2.len(), 6, "thin=2: 2 chains x 3 sampled iters");
    }

    #[test]
    fn phyclone_max_time_stops_trace_early() {
        use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};

        let clusters = vec![CompatClusterSummary {
            cluster_id: 0,
            mutation_ids: vec!["m0".to_string()],
            sample_ids: vec!["s0".to_string()],
            sample_profiles: Vec::new(),
            outlier_data_points: Vec::new(),
        }];

        let data_points = vec![CompatDataPoint {
            idx: 0,
            name: CompatDataPointName::Int(0),
            mutation_ids: vec!["m0".to_string()],
            sample_ids: vec!["s0".to_string()],
            value: vec![vec![-1.2, -0.8, -0.4]],
            raw_outlier_prob: 0.01,
            outlier_prob: (0.01_f64).ln(),
            outlier_prob_not: (0.99_f64).ln(),
            outlier_marginal_prob: (0.01_f64).ln(),
            size: 1,
        }];

        let records = run_phyclone_mcmc_traces_from_data_points(
            1,
            6,
            2,
            1,
            &data_points,
            &clusters,
            Some(7),
            4,
            0.5,
            0.0,
            100,
            ProposalFamily::FullyAdapted,
            1,
            1,
            false,
            1.0,
            0.0,
        )
        .expect("phyclone trace should succeed");

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].iter, 0);
    }

    #[test]
    fn phyclone_trace_uses_stored_scores_not_recomputed_final_joint() {
        use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};
        use crate::phyclone::compat::smc::CompatMcmcSample;
        use crate::phyclone::compat::tree_model::CompatTree;

        let clusters = vec![CompatClusterSummary {
            cluster_id: 0,
            mutation_ids: vec!["m0".to_string()],
            sample_ids: vec!["s0".to_string()],
            sample_profiles: Vec::new(),
            outlier_data_points: Vec::new(),
        }];
        let data_points = vec![CompatDataPoint {
            idx: 0,
            name: CompatDataPointName::Int(0),
            mutation_ids: vec!["m0".to_string()],
            sample_ids: vec!["s0".to_string()],
            value: vec![vec![-1.2, -0.8, -0.4]],
            raw_outlier_prob: 0.01,
            outlier_prob: (0.01_f64).ln(),
            outlier_prob_not: (0.99_f64).ln(),
            outlier_marginal_prob: (0.01_f64).ln(),
            size: 1,
        }];

        let mut tree = CompatTree::new("root", 1, 3, -(3_f64).ln(), 8);
        tree.add_child_node("root", "n0").expect("add n0");
        tree.node_mut("n0")
            .expect("n0 exists")
            .add_data_point(0, &data_points[0].value)
            .expect("add datapoint");
        tree.update_path_to_root("n0").expect("update path");

        let sample = CompatMcmcSample {
            iter: 4,
            tree,
            alpha: 0.5,
            log_p: -123.0,
            log_p_one: -45.0,
        };

        let record =
            mcmc_sample_to_trace_record(&sample, &data_points, &clusters, 0, Some(7), 1, 10);

        assert_eq!(record.iter, 4);
        assert_eq!(record.log_p, -123.0);
        assert_eq!(record.log_p_one, -45.0);
    }

    /// Iterations within the burn-in period are excluded from the output.
    #[test]
    fn burnin_excluded_from_output() {
        use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};

        let clusters = vec![CompatClusterSummary {
            cluster_id: 0,
            mutation_ids: vec!["m0".to_string()],
            sample_ids: vec!["s0".to_string()],
            sample_profiles: Vec::new(),
            outlier_data_points: Vec::new(),
        }];
        let data_points: Vec<CompatDataPoint> = vec![CompatDataPoint {
            idx: 0,
            name: CompatDataPointName::Int(0),
            mutation_ids: vec!["m0".to_string()],
            sample_ids: vec!["s0".to_string()],
            value: vec![vec![-1.2, -0.8, -0.4]],
            raw_outlier_prob: 0.01,
            outlier_prob: (0.01_f64).ln(),
            outlier_prob_not: (0.99_f64).ln(),
            outlier_marginal_prob: (0.01_f64).ln(),
            size: 1,
        }];

        // num_iters=4, burnin=2 -> 2 records per chain
        let with_burnin = run_compat_mcmc_traces_from_data_points(
            1,
            4,
            2,
            1,
            &data_points,
            &clusters,
            Some(42),
            4,
            0.5,
        );
        // num_iters=4, burnin=0 -> 4 records per chain
        let no_burnin = run_compat_mcmc_traces_from_data_points(
            1,
            4,
            0,
            1,
            &data_points,
            &clusters,
            Some(42),
            4,
            0.5,
        );
        assert_eq!(no_burnin.len(), 4, "burnin=0: all 4 iters kept");
        assert_eq!(
            with_burnin.len(),
            2,
            "burnin=2: only 2 post-burnin iters kept"
        );
    }

    /// Fixing the seed produces identical traces across runs (multi-chain).
    #[test]
    fn seed_reproducibility_multi_chain() {
        use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};

        let clusters = vec![
            CompatClusterSummary {
                cluster_id: 0,
                mutation_ids: vec!["m0".to_string()],
                sample_ids: vec!["s0".to_string()],
                sample_profiles: Vec::new(),
                outlier_data_points: Vec::new(),
            },
            CompatClusterSummary {
                cluster_id: 1,
                mutation_ids: vec!["m1".to_string()],
                sample_ids: vec!["s0".to_string()],
                sample_profiles: Vec::new(),
                outlier_data_points: Vec::new(),
            },
        ];
        let data_points: Vec<CompatDataPoint> = (0..2)
            .map(|i| CompatDataPoint {
                idx: i,
                name: CompatDataPointName::Int(i as i64),
                mutation_ids: vec![format!("m{}", i)],
                sample_ids: vec!["s0".to_string()],
                value: vec![vec![-1.2, -0.8, -0.4]],
                raw_outlier_prob: 0.01,
                outlier_prob: (0.01_f64).ln(),
                outlier_prob_not: (0.99_f64).ln(),
                outlier_marginal_prob: (0.01_f64).ln(),
                size: 1,
            })
            .collect();

        let run1 = run_compat_mcmc_traces_from_data_points(
            2,
            4,
            1,
            1,
            &data_points,
            &clusters,
            Some(99),
            8,
            0.5,
        );
        let run2 = run_compat_mcmc_traces_from_data_points(
            2,
            4,
            1,
            1,
            &data_points,
            &clusters,
            Some(99),
            8,
            0.5,
        );

        assert_eq!(run1.len(), run2.len());
        for (a, b) in run1.iter().zip(run2.iter()) {
            assert_eq!(a.chain, b.chain);
            assert_eq!(a.iter, b.iter);
            assert_eq!(
                (a.log_p * 1e6).round(),
                (b.log_p * 1e6).round(),
                "log_p mismatch at chain={} iter={}",
                a.chain,
                a.iter
            );
            assert_eq!(a.topology_id, b.topology_id);
        }
    }
}
