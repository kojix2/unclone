#![allow(dead_code)]

use super::convolution::{compute_log_s_1d_with_cache, ConvolutionCache};
use super::tree_ids::{DataPointId, NodeId};

/// A single node in a `CompatTree`.
///
/// `log_p` and `log_r` are **samples x grid** matrices, matching PhyClone's
/// `TreeNode.log_p` / `TreeNode.log_r` which are 2-D NumPy arrays of shape
/// `(num_samples, grid_size)`.
#[derive(Clone, Debug, PartialEq)]
pub struct CompatTreeNode {
    pub node_id: NodeId,
    /// log P(data assigned to this node) per sample x grid cell.
    pub log_p: Vec<Vec<f64>>,
    /// log R (subtree sufficient statistic) per sample x grid cell.
    pub log_r: Vec<Vec<f64>>,
    pub data_point_ids: Vec<DataPointId>,
}

impl CompatTreeNode {
    pub fn new(
        node_id: impl Into<NodeId>,
        num_samples: usize,
        grid_size: usize,
        log_prior: f64,
    ) -> Self {
        Self {
            node_id: node_id.into(),
            log_p: vec![vec![log_prior; grid_size]; num_samples],
            log_r: vec![vec![0.0; grid_size]; num_samples],
            data_point_ids: Vec::new(),
        }
    }

    /// Add a data point to this node.
    ///
    /// `value` must be `samples x grid`, matching `CompatDataPoint::value`.
    pub fn add_data_point(
        &mut self,
        data_point_id: DataPointId,
        value: &[Vec<f64>],
    ) -> Result<(), String> {
        if value.len() != self.log_p.len() {
            return Err("data point sample count must match node sample count".to_string());
        }
        for (s, sample_vals) in value.iter().enumerate() {
            if sample_vals.len() != self.log_p[s].len() {
                return Err("data point grid size must match node grid size".to_string());
            }
        }
        if self.data_point_ids.contains(&data_point_id) {
            return Err("duplicate data point id in node".to_string());
        }
        self.data_point_ids.push(data_point_id);
        for (s, sample_vals) in value.iter().enumerate() {
            for (g, &v) in sample_vals.iter().enumerate() {
                self.log_p[s][g] += v;
                self.log_r[s][g] += v;
            }
        }
        Ok(())
    }

    /// Remove a previously added data point from this node.
    pub fn remove_data_point(
        &mut self,
        data_point_id: DataPointId,
        value: &[Vec<f64>],
    ) -> Result<(), String> {
        if value.len() != self.log_p.len() {
            return Err("data point sample count must match node sample count".to_string());
        }
        let Some(pos) = self
            .data_point_ids
            .iter()
            .position(|id| *id == data_point_id)
        else {
            return Err("data point id not found in node".to_string());
        };
        self.data_point_ids.swap_remove(pos);
        for (s, sample_vals) in value.iter().enumerate() {
            for (g, &v) in sample_vals.iter().enumerate() {
                self.log_p[s][g] -= v;
            }
        }
        Ok(())
    }

    pub fn update_node_from_child_r_vals(
        &mut self,
        child_log_r_values: &[Vec<Vec<f64>>],
    ) -> Result<(), String> {
        self.update_node_from_child_r_vals_with_cache(child_log_r_values, None)
    }

    /// Update `log_r` from children's `log_r` values.
    ///
    /// `child_log_r_values` is a slice over children; each element is `samples x grid`.
    /// Per-sample 1-D convolution is performed independently, then combined with `log_p`.
    pub fn update_node_from_child_r_vals_with_cache(
        &mut self,
        child_log_r_values: &[Vec<Vec<f64>>],
        mut cache: Option<&mut ConvolutionCache>,
    ) -> Result<(), String> {
        if child_log_r_values.is_empty() {
            self.log_r.clone_from(&self.log_p);
            return Ok(());
        }

        let num_samples = self.log_p.len();
        for child in child_log_r_values {
            if child.len() != num_samples {
                return Err("child log_R sample count must match node sample count".to_string());
            }
        }

        for s in 0..num_samples {
            let grid_size = self.log_p[s].len();
            let per_sample: Vec<Vec<f64>> =
                child_log_r_values.iter().map(|c| c[s].clone()).collect();
            for child_sample in &per_sample {
                if child_sample.len() != grid_size {
                    return Err("child log_R grid size must match node grid size".to_string());
                }
            }
            let log_s = compute_log_s_1d_with_cache(&per_sample, cache.as_deref_mut())?;
            for (g, ls) in log_s.iter().enumerate().take(grid_size) {
                self.log_r[s][g] = self.log_p[s][g] + ls;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::CompatTreeNode;
    use crate::phyclone::compat::convolution::ConvolutionCache;

    fn lse(values: &[f64]) -> f64 {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max.is_infinite() && max.is_sign_negative() {
            return max;
        }
        let sum: f64 = values.iter().map(|v| (*v - max).exp()).sum();
        max + sum.ln()
    }

    #[test]
    fn update_without_children_copies_log_p_to_log_r() {
        let mut node = CompatTreeNode::new("n0", 1, 3, -1.0);
        node.log_p = vec![vec![-1.2, -0.2, -3.4]];
        node.log_r = vec![vec![0.0, 0.0, 0.0]];

        node.update_node_from_child_r_vals(&[])
            .expect("no-child update should succeed");

        assert_eq!(node.log_r, node.log_p);
    }

    #[test]
    fn update_with_children_matches_manual_log_p_plus_log_s() {
        let mut node = CompatTreeNode::new("n1", 1, 3, -1.0);
        node.log_p = vec![vec![-0.5, -0.4, -1.0]];

        // children are samples x grid (1 sample, 3 grid points)
        let child_a = vec![vec![-0.2, -1.1, -0.7]];
        let child_b = vec![vec![-1.2, -0.3, -2.0]];

        node.update_node_from_child_r_vals(&[child_a.clone(), child_b.clone()])
            .expect("child update should succeed");

        let log_d_0 = child_a[0][0] + child_b[0][0];
        let log_d_1 = lse(&[child_a[0][0] + child_b[0][1], child_a[0][1] + child_b[0][0]]);
        let log_d_2 = lse(&[
            child_a[0][0] + child_b[0][2],
            child_a[0][1] + child_b[0][1],
            child_a[0][2] + child_b[0][0],
        ]);

        let log_s_0 = log_d_0;
        let log_s_1 = lse(&[log_d_0, log_d_1]);
        let log_s_2 = lse(&[log_d_0, log_d_1, log_d_2]);

        let expected = vec![
            node.log_p[0][0] + log_s_0,
            node.log_p[0][1] + log_s_1,
            node.log_p[0][2] + log_s_2,
        ];

        for (a, e) in node.log_r[0].iter().zip(expected.iter()) {
            assert!((a - e).abs() <= 1e-10);
        }
    }

    #[test]
    fn update_with_cache_matches_update_without_cache() {
        let mut no_cache = CompatTreeNode::new("n2", 1, 3, -1.0);
        no_cache.log_p = vec![vec![-0.5, -0.4, -1.0]];

        let mut with_cache = CompatTreeNode::new("n2", 1, 3, -1.0);
        with_cache.log_p = vec![vec![-0.5, -0.4, -1.0]];

        // children: Vec<Vec<Vec<f64>>> = [child x sample x grid]
        let children: Vec<Vec<Vec<f64>>> = vec![
            vec![vec![-0.2, -1.1, -0.7]],
            vec![vec![-1.2, -0.3, -2.0]],
            vec![vec![-0.6, -0.7, -1.5]],
        ];

        no_cache
            .update_node_from_child_r_vals(&children)
            .expect("no-cache update should succeed");

        let mut cache = ConvolutionCache::new(8);
        with_cache
            .update_node_from_child_r_vals_with_cache(&children, Some(&mut cache))
            .expect("cached update should succeed");

        assert_eq!(no_cache.log_r.len(), with_cache.log_r.len());
        for (a_sample, e_sample) in with_cache.log_r.iter().zip(no_cache.log_r.iter()) {
            for (a, e) in a_sample.iter().zip(e_sample.iter()) {
                assert!((a - e).abs() <= 1e-10);
            }
        }
    }

    #[test]
    fn update_errors_on_child_grid_mismatch() {
        let mut node = CompatTreeNode::new("n3", 1, 3, -1.0);
        let err = node
            .update_node_from_child_r_vals(&[vec![vec![-1.0, -2.0]]])
            .expect_err("grid mismatch must error");
        assert!(err.contains("grid size"));
    }
}
