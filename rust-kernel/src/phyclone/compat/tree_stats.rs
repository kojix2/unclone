#![allow(dead_code)]

use super::tree_ids::DataPointId;

/// Cluster assignment info for a single non-root node.
#[derive(Clone, Debug, PartialEq)]
pub struct CompatNodeCluster {
    pub data_point_count: usize,
    pub is_outlier_node: bool,
}

/// Topology-derived statistics used by the tree prior distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct CompatTreePriorStats {
    pub num_nodes: usize,
    pub multiplicity_log: f64,
    pub root_subtree_node_counts: Vec<usize>,
}

/// Per-data-point outlier probability triple.
#[derive(Clone, Debug, PartialEq)]
pub struct CompatOutlierPoint {
    pub id: DataPointId,
    pub outlier_prob: f64,
    pub outlier_prob_not: f64,
    pub outlier_marginal_prob: f64,
}

/// Tree data log-likelihood summary returned by `CompatTree::tree_likelihood`.
///
/// `data_log_likelihood[s][g]` is the root node's log-R value for sample `s`
/// at CCF grid index `g`.  This is the upstream-compatible representation used
/// by `CompatTreeJointDistribution`.
#[derive(Clone, Debug, PartialEq)]
pub struct CompatTreeLikelihood {
    pub root_children_count: usize,
    pub data_log_likelihood: Vec<Vec<f64>>, // samples x grid
}
