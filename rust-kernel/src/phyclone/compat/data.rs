#![allow(dead_code)]

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatDataPointName {
    Int(i64),
    Str(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatDataPoint {
    pub idx: usize,
    pub name: CompatDataPointName,
    /// Mutation IDs that were grouped into this cluster data point by the loader.
    /// Only contains mutations that passed PhyClone-style filtering.
    pub mutation_ids: Vec<String>,
    /// Sample IDs present for this data point, in sorted order.
    pub sample_ids: Vec<String>,
    pub value: Vec<Vec<f64>>,
    /// Raw, non-log outlier probability used to decide whether outlier modelling
    /// should be active.
    #[serde(default)]
    pub raw_outlier_prob: f64,
    pub outlier_prob: f64,
    pub outlier_prob_not: f64,
    pub outlier_marginal_prob: f64,
    pub size: usize,
}

impl CompatDataPoint {
    /// Returns the marginal log-likelihood vector for this data point by summing the
    /// per-sample log-likelihood grids across all samples.
    ///
    /// `result[grid_idx] = sum_s value[s][grid_idx]`
    ///
    /// This is the value that should be added to a `CompatTreeNode` when the data
    /// point is assigned to that node.
    pub fn marginal_log_likelihood(&self) -> Vec<f64> {
        if self.value.is_empty() {
            return Vec::new();
        }
        let grid_size = self.value[0].len();
        let mut marginal = vec![0.0_f64; grid_size];
        for sample_ll in &self.value {
            for (idx, v) in sample_ll.iter().enumerate() {
                marginal[idx] += v;
            }
        }
        marginal
    }
}
