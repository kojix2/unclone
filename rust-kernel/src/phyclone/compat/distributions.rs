#![allow(dead_code)]

use super::tree_ids::DataPointId;
use super::tree_model::CompatTree;
use super::tree_stats::{
    CompatNodeCluster, CompatOutlierPoint, CompatTreeLikelihood, CompatTreePriorStats,
};
use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq)]
pub struct CompatFscrpDistribution {
    pub alpha: f64,
    pub c_const: f64,
}

impl Default for CompatFscrpDistribution {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            c_const: 1000.0,
        }
    }
}

impl CompatFscrpDistribution {
    pub fn compute_crp_prior(
        &self,
        node_clusters: &[CompatNodeCluster],
        num_nodes: usize,
    ) -> Result<f64, String> {
        if num_nodes == 0 {
            return Ok(0.0);
        }
        if node_clusters.is_empty() {
            return Err("node clusters must not be empty when num_nodes > 0".to_string());
        }

        let alpha = if self.alpha == 0.0 {
            f64::MIN_POSITIVE
        } else {
            self.alpha.max(f64::MIN_POSITIVE)
        };

        let mut log_p = (num_nodes as f64) * alpha.ln();

        for cluster in node_clusters {
            if cluster.is_outlier_node || cluster.data_point_count == 0 {
                continue;
            }
            log_p += log_factorial(cluster.data_point_count - 1);
        }

        Ok(log_p)
    }

    pub fn log_p(
        &self,
        stats: &CompatTreePriorStats,
        node_clusters: &[CompatNodeCluster],
        crp_prior: Option<f64>,
    ) -> Result<f64, String> {
        let mut log_p = match crp_prior {
            Some(value) => value,
            None => self.compute_crp_prior(node_clusters, stats.num_nodes)?,
        };

        if stats.num_nodes > 0 {
            log_p -= ((stats.num_nodes - 1) as f64) * ((stats.num_nodes + 1) as f64).ln();
        }
        log_p -= stats.multiplicity_log;

        Ok(log_p)
    }

    pub fn log_p_one(
        &self,
        stats: &CompatTreePriorStats,
        node_clusters: &[CompatNodeCluster],
        crp_prior: Option<f64>,
    ) -> Result<f64, String> {
        let mut log_p = match crp_prior {
            Some(value) => value,
            None => self.compute_crp_prior(node_clusters, stats.num_nodes)?,
        };

        let num_roots = stats.root_subtree_node_counts.len();
        let r_term = self.compute_r_term(num_roots, stats.num_nodes);

        let mut num_ways = 0.0;
        for &curr_num_nodes in &stats.root_subtree_node_counts {
            if curr_num_nodes > 1 {
                num_ways += ((curr_num_nodes - 1) as f64) * (curr_num_nodes as f64).ln();
            }
        }

        log_p += -num_ways + r_term;
        log_p -= stats.multiplicity_log;

        Ok(log_p)
    }

    pub fn compute_z_term(&self, num_roots: usize, num_nodes: usize) -> f64 {
        let log_one = 0.0;
        let a_term = log_one * (num_nodes as f64);

        if num_roots == 0 {
            return a_term;
        }

        let log_const = self.c_const.max(f64::MIN_POSITIVE).ln();
        let la = log_one;

        let mut numerator = log_one - (log_const * (num_roots as f64));
        let mut denominator = log_one - log_const;

        numerator = la + (1.0 - (numerator - la).exp()).ln();
        denominator = la + (1.0 - (denominator - la).exp()).ln();

        a_term + (numerator - denominator)
    }

    pub fn compute_r_term(&self, num_roots: usize, num_nodes: usize) -> f64 {
        let z_term = self.compute_z_term(num_roots, num_nodes);
        let log_const = self.c_const.max(f64::MIN_POSITIVE).ln();

        let adjusted_roots = if num_roots == 0 { 1 } else { num_roots };
        -(z_term + (log_const * ((adjusted_roots - 1) as f64)))
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct CompatTreeJointDistribution {
    pub prior: CompatFscrpDistribution,
    pub outlier_modelling_active: bool,
}

impl CompatTreeJointDistribution {
    pub fn log_p(
        &self,
        stats: &CompatTreePriorStats,
        node_clusters: &[CompatNodeCluster],
        likelihood: Option<&CompatTreeLikelihood>,
        outlier_points: &[CompatOutlierPoint],
        assigned_outliers: &HashSet<DataPointId>,
    ) -> Result<f64, String> {
        let mut score = self.prior.log_p(stats, node_clusters, None)?;
        score += self.outlier_prior(outlier_points, assigned_outliers);

        if let Some(likelihood) = likelihood {
            score += self.tree_marginal_log_likelihood(likelihood)?;
        }

        score += self.outlier_marginal_prob(outlier_points, assigned_outliers);
        Ok(score)
    }

    pub fn log_p_one(
        &self,
        stats: &CompatTreePriorStats,
        node_clusters: &[CompatNodeCluster],
        likelihood: Option<&CompatTreeLikelihood>,
        outlier_points: &[CompatOutlierPoint],
        assigned_outliers: &HashSet<DataPointId>,
    ) -> Result<f64, String> {
        let mut score = self.prior.log_p_one(stats, node_clusters, None)?;
        score += self.outlier_prior(outlier_points, assigned_outliers);

        if let Some(likelihood) = likelihood {
            score += self.tree_log_likelihood_at_one(likelihood)?;
        }

        score += self.outlier_marginal_prob(outlier_points, assigned_outliers);
        Ok(score)
    }

    pub fn compute_log_p_and_log_p_one(
        &self,
        stats: &CompatTreePriorStats,
        node_clusters: &[CompatNodeCluster],
        likelihood: Option<&CompatTreeLikelihood>,
        outlier_points: &[CompatOutlierPoint],
        assigned_outliers: &HashSet<DataPointId>,
    ) -> Result<(f64, f64), String> {
        let crp_prior = self
            .prior
            .compute_crp_prior(node_clusters, stats.num_nodes)?;

        let mut log_p = self.prior.log_p(stats, node_clusters, Some(crp_prior))?;
        let mut log_p_one = self
            .prior
            .log_p_one(stats, node_clusters, Some(crp_prior))?;

        let outlier_prior = self.outlier_prior(outlier_points, assigned_outliers);
        log_p += outlier_prior;
        log_p_one += outlier_prior;

        if let Some(likelihood) = likelihood {
            log_p += self.tree_marginal_log_likelihood(likelihood)?;
            log_p_one += self.tree_log_likelihood_at_one(likelihood)?;
        }

        let outlier_marginal = self.outlier_marginal_prob(outlier_points, assigned_outliers);
        log_p += outlier_marginal;
        log_p_one += outlier_marginal;

        Ok((log_p, log_p_one))
    }

    /// Compute the root permutation distribution log-pdf.
    /// Compute the log-pdf of the root permutation distribution for the given tree.
    ///
    /// Mirrors `RootPermutationDistribution.log_pdf(tree)` from PhyClone's `smc/utils.py`:
    /// ```python
    /// return -RootPermutationDistribution.log_count(tree)
    /// ```
    ///
    /// This is a combinatorial term counting the number of valid data orderings consistent
    /// with the tree topology. It is NOT `log_p_one - log_p`.
    pub fn log_pdf_root_permutation(&self, tree: &super::tree_model::CompatTree) -> f64 {
        super::root_permutation::RootPermutationDistribution::log_pdf(tree)
    }

    pub fn outlier_prior(
        &self,
        outlier_points: &[CompatOutlierPoint],
        assigned_outliers: &HashSet<DataPointId>,
    ) -> f64 {
        if !self.outlier_modelling_active {
            return 0.0;
        }

        outlier_points
            .iter()
            .map(|point| {
                if assigned_outliers.contains(&point.id) {
                    point.outlier_prob
                } else {
                    point.outlier_prob_not
                }
            })
            .sum()
    }

    pub fn outlier_marginal_prob(
        &self,
        outlier_points: &[CompatOutlierPoint],
        assigned_outliers: &HashSet<DataPointId>,
    ) -> f64 {
        if !self.outlier_modelling_active {
            return 0.0;
        }

        outlier_points
            .iter()
            .map(|point| {
                if assigned_outliers.contains(&point.id) {
                    point.outlier_marginal_prob
                } else {
                    0.0
                }
            })
            .sum()
    }

    fn tree_marginal_log_likelihood(
        &self,
        likelihood: &CompatTreeLikelihood,
    ) -> Result<f64, String> {
        if likelihood.root_children_count == 0 {
            return Ok(0.0);
        }
        log_sum_exp_over_dims(&likelihood.data_log_likelihood)
    }

    fn tree_log_likelihood_at_one(&self, likelihood: &CompatTreeLikelihood) -> Result<f64, String> {
        if likelihood.root_children_count == 0 {
            return Ok(0.0);
        }

        let mut total = 0.0;
        for row in &likelihood.data_log_likelihood {
            let Some(last) = row.last() else {
                return Err("likelihood row must not be empty".to_string());
            };
            total += *last;
        }
        Ok(total)
    }

    /// Compute log p(tree, data) from a `CompatTree`, including outlier modelling.
    ///
    /// Uses `tree.assigned_outliers` and the `outlier_points` derived from the tree's
    /// data points.  Pass an empty slice for `outlier_points` when outlier modelling
    /// is not needed.
    pub fn log_p_tree(
        &self,
        tree: &CompatTree,
        outlier_points: &[CompatOutlierPoint],
    ) -> Result<f64, String> {
        let likelihood = tree.tree_likelihood()?;
        let stats = tree.prior_stats()?;
        let clusters = tree.node_clusters();
        self.log_p(
            &stats,
            &clusters,
            Some(&likelihood),
            outlier_points,
            &tree.assigned_outliers,
        )
    }

    /// Compute log p_one(tree, data) (the fixed-point variant) from a `CompatTree`.
    ///
    /// Uses CCF=1 for the integration bound, with outlier modelling.
    pub fn log_p_one_tree(
        &self,
        tree: &CompatTree,
        outlier_points: &[CompatOutlierPoint],
    ) -> Result<f64, String> {
        let likelihood = tree.tree_likelihood()?;
        let stats = tree.prior_stats()?;
        let clusters = tree.node_clusters();
        self.log_p_one(
            &stats,
            &clusters,
            Some(&likelihood),
            outlier_points,
            &tree.assigned_outliers,
        )
    }

    /// Compute both log p and log p_one in a single pass for efficiency.
    pub fn compute_log_p_and_log_p_one_tree(
        &self,
        tree: &CompatTree,
        outlier_points: &[CompatOutlierPoint],
    ) -> Result<(f64, f64), String> {
        let likelihood = tree.tree_likelihood()?;
        let stats = tree.prior_stats()?;
        let clusters = tree.node_clusters();
        self.compute_log_p_and_log_p_one(
            &stats,
            &clusters,
            Some(&likelihood),
            outlier_points,
            &tree.assigned_outliers,
        )
    }

    /// Create a scorer closure suitable for passing to proposal set builders.
    ///
    /// The returned closure captures `data_points` by `Arc` and calls `log_p_tree` on each
    /// candidate tree.  This satisfies the `F: Fn(&CompatTree) -> f64` bound required by
    /// `build_fully_adapted_proposal_set` / `build_semi_adapted_proposal_set`.
    ///
    /// # Example
    /// ```ignore
    /// let scorer = joint.make_log_p_scorer(Arc::new(data_points.to_vec()));
    /// let set = build_fully_adapted_proposal_set(&tree, &adder, dp_id, &dp_value, true, scorer)?;
    /// ```
    pub fn make_log_p_scorer(
        &self,
        data_points: std::sync::Arc<Vec<super::data::CompatDataPoint>>,
    ) -> impl Fn(&CompatTree) -> f64 + '_ {
        move |tree: &CompatTree| {
            let outlier_pts = CompatTree::outlier_points(&data_points);
            self.log_p_tree(tree, &outlier_pts)
                .unwrap_or(f64::NEG_INFINITY)
        }
    }
}

fn log_factorial(n: usize) -> f64 {
    if n < 2 {
        return 0.0;
    }
    (2..=n).fold(0.0, |acc, x| acc + (x as f64).ln())
}

fn log_sum_exp(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("values must not be empty".to_string());
    }

    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() && max.is_sign_negative() {
        return Ok(max);
    }

    let sum: f64 = values.iter().map(|v| (*v - max).exp()).sum();
    Ok(max + sum.ln())
}

fn log_sum_exp_over_dims(matrix: &[Vec<f64>]) -> Result<f64, String> {
    let mut total = 0.0;
    for row in matrix {
        total += log_sum_exp(row)?;
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phyclone::compat::tree_stats::*;
    use serde::Deserialize;
    use std::fs;
    use std::path::Path;

    fn approx_eq(lhs: f64, rhs: f64) {
        let delta = (lhs - rhs).abs();
        assert!(delta <= 1e-10, "lhs={} rhs={} delta={}", lhs, rhs, delta);
    }

    fn approx_eq_tol(lhs: f64, rhs: f64, tol: f64) {
        let delta = (lhs - rhs).abs();
        assert!(
            delta <= tol,
            "lhs={} rhs={} delta={} tol={}",
            lhs,
            rhs,
            delta,
            tol
        );
    }

    #[derive(Debug, Deserialize)]
    struct FixedTreeOracleFixture {
        fixed_tree_oracle: FixedTreeOracle,
    }

    #[derive(Debug, Deserialize)]
    struct FixedTreeOracle {
        prior: FixedTreePriorConfig,
        cases: Vec<FixedTreeCase>,
    }

    #[derive(Debug, Deserialize)]
    struct FixedTreePriorConfig {
        alpha: f64,
        c_const: f64,
    }

    #[derive(Debug, Deserialize)]
    struct FixedTreeCase {
        name: String,
        tree_prior_stats: FixedTreePriorStats,
        node_clusters: Vec<FixedNodeCluster>,
        likelihood: FixedTreeLikelihood,
        outlier_points: Vec<FixedOutlierPoint>,
        assigned_outlier_ids: Vec<usize>,
        expected: FixedTreeExpected,
    }

    #[derive(Debug, Deserialize)]
    struct FixedTreePriorStats {
        num_nodes: usize,
        multiplicity_log: f64,
        root_subtree_node_counts: Vec<usize>,
    }

    #[derive(Debug, Deserialize)]
    struct FixedNodeCluster {
        data_point_count: usize,
        is_outlier_node: bool,
    }

    #[derive(Debug, Deserialize)]
    struct FixedTreeLikelihood {
        root_children_count: usize,
        data_log_likelihood: Vec<Vec<f64>>,
    }

    #[derive(Debug, Deserialize)]
    struct FixedOutlierPoint {
        id: usize,
        outlier_prob: f64,
        outlier_prob_not: f64,
        outlier_marginal_prob: f64,
    }

    #[derive(Debug, Deserialize)]
    struct FixedTreeExpected {
        crp_prior: f64,
        log_p: f64,
        log_p_one: f64,
    }

    #[test]
    fn crp_prior_matches_manual_formula() {
        let fscrp = CompatFscrpDistribution {
            alpha: 2.0,
            c_const: 1000.0,
        };
        let clusters = vec![
            CompatNodeCluster {
                data_point_count: 3,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 2,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 4,
                is_outlier_node: true,
            },
        ];

        let actual = fscrp
            .compute_crp_prior(&clusters, 3)
            .expect("crp prior should compute");
        let expected = 3.0 * 2.0_f64.ln() + log_factorial(2) + log_factorial(1);

        approx_eq(actual, expected);
    }

    #[test]
    fn log_p_and_log_p_one_include_r_and_multiplicity_terms() {
        let fscrp = CompatFscrpDistribution::default();
        let stats = CompatTreePriorStats {
            num_nodes: 4,
            multiplicity_log: 1.2,
            root_subtree_node_counts: vec![3, 1],
        };
        let clusters = vec![
            CompatNodeCluster {
                data_point_count: 2,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 1,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 3,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 0,
                is_outlier_node: true,
            },
        ];

        let crp = fscrp
            .compute_crp_prior(&clusters, stats.num_nodes)
            .expect("crp prior should compute");

        let log_p = fscrp
            .log_p(&stats, &clusters, Some(crp))
            .expect("log_p should compute");
        let log_p_one = fscrp
            .log_p_one(&stats, &clusters, Some(crp))
            .expect("log_p_one should compute");

        assert!(log_p.is_finite());
        assert!(log_p_one.is_finite());
        assert!(log_p_one <= log_p + 50.0);
    }

    #[test]
    fn joint_distribution_adds_outlier_and_likelihood_terms() {
        let joint = CompatTreeJointDistribution {
            prior: CompatFscrpDistribution::default(),
            outlier_modelling_active: true,
        };

        let stats = CompatTreePriorStats {
            num_nodes: 3,
            multiplicity_log: 0.5,
            root_subtree_node_counts: vec![3],
        };
        let clusters = vec![
            CompatNodeCluster {
                data_point_count: 2,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 1,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 0,
                is_outlier_node: true,
            },
        ];
        let likelihood = CompatTreeLikelihood {
            root_children_count: 1,
            data_log_likelihood: vec![vec![-2.0, -1.0, -0.7], vec![-1.5, -0.8, -0.2]],
        };
        let outlier_points = vec![
            CompatOutlierPoint {
                id: 10,
                outlier_prob: -2.0,
                outlier_prob_not: -0.1,
                outlier_marginal_prob: -0.4,
            },
            CompatOutlierPoint {
                id: 11,
                outlier_prob: -3.0,
                outlier_prob_not: -0.2,
                outlier_marginal_prob: -0.5,
            },
        ];
        let assigned: HashSet<DataPointId> = [10].into_iter().collect();

        let (log_p, log_p_one) = joint
            .compute_log_p_and_log_p_one(
                &stats,
                &clusters,
                Some(&likelihood),
                &outlier_points,
                &assigned,
            )
            .expect("joint scores should compute");

        assert!(log_p.is_finite());
        assert!(log_p_one.is_finite());

        let outlier_prior = joint.outlier_prior(&outlier_points, &assigned);
        approx_eq(outlier_prior, -2.2);

        let outlier_marginal = joint.outlier_marginal_prob(&outlier_points, &assigned);
        approx_eq(outlier_marginal, -0.4);
    }

    #[test]
    fn compute_log_p_and_log_p_one_matches_individual_methods() {
        let joint = CompatTreeJointDistribution {
            prior: CompatFscrpDistribution {
                alpha: 1.5,
                c_const: 500.0,
            },
            outlier_modelling_active: true,
        };

        let stats = CompatTreePriorStats {
            num_nodes: 2,
            multiplicity_log: 0.3,
            root_subtree_node_counts: vec![2],
        };
        let clusters = vec![
            CompatNodeCluster {
                data_point_count: 1,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 0,
                is_outlier_node: true,
            },
        ];
        let likelihood = CompatTreeLikelihood {
            root_children_count: 1,
            data_log_likelihood: vec![vec![-1.0, -0.2]],
        };
        let outlier_points = vec![CompatOutlierPoint {
            id: 5,
            outlier_prob: -1.2,
            outlier_prob_not: -0.4,
            outlier_marginal_prob: -0.6,
        }];
        let assigned: HashSet<DataPointId> = [5].into_iter().collect();

        let pair = joint
            .compute_log_p_and_log_p_one(
                &stats,
                &clusters,
                Some(&likelihood),
                &outlier_points,
                &assigned,
            )
            .expect("pair scores should compute");

        let log_p = joint
            .log_p(
                &stats,
                &clusters,
                Some(&likelihood),
                &outlier_points,
                &assigned,
            )
            .expect("log_p should compute");
        let log_p_one = joint
            .log_p_one(
                &stats,
                &clusters,
                Some(&likelihood),
                &outlier_points,
                &assigned,
            )
            .expect("log_p_one should compute");

        approx_eq(pair.0, log_p);
        approx_eq(pair.1, log_p_one);
    }

    #[test]
    fn matches_python_fixed_tree_oracle_fixture() {
        let path_str = "src/phyclone/compat/testdata/phyclone_oracle_fixed_tree_scores.json";
        let path = Path::new(path_str);
        if !path.exists() {
            eprintln!(
                "skipping fixed-tree oracle parity test: fixture not found at {}",
                path_str
            );
            return;
        }

        let contents = fs::read_to_string(path).expect("fixture read should succeed");
        let fixture: FixedTreeOracleFixture = match serde_json::from_str(&contents) {
            Ok(f) => f,
            Err(e) => {
                eprintln!(
                    "skipping fixed-tree oracle parity test: fixture schema mismatch ({}). \
                     Populate {} with the correct 'fixed_tree_oracle' key to enable this test.",
                    e, path_str
                );
                return;
            }
        };

        let prior = CompatFscrpDistribution {
            alpha: fixture.fixed_tree_oracle.prior.alpha,
            c_const: fixture.fixed_tree_oracle.prior.c_const,
        };
        let joint = CompatTreeJointDistribution {
            prior: prior.clone(),
            outlier_modelling_active: true,
        };

        for case in fixture.fixed_tree_oracle.cases {
            let assigned: HashSet<DataPointId> = case.assigned_outlier_ids.into_iter().collect();
            let stats = CompatTreePriorStats {
                num_nodes: case.tree_prior_stats.num_nodes,
                multiplicity_log: case.tree_prior_stats.multiplicity_log,
                root_subtree_node_counts: case.tree_prior_stats.root_subtree_node_counts,
            };
            let clusters: Vec<CompatNodeCluster> = case
                .node_clusters
                .into_iter()
                .map(|c| CompatNodeCluster {
                    data_point_count: c.data_point_count,
                    is_outlier_node: c.is_outlier_node,
                })
                .collect();
            let likelihood = CompatTreeLikelihood {
                root_children_count: case.likelihood.root_children_count,
                data_log_likelihood: case.likelihood.data_log_likelihood,
            };
            let outlier_points: Vec<CompatOutlierPoint> = case
                .outlier_points
                .into_iter()
                .map(|p| CompatOutlierPoint {
                    id: p.id,
                    outlier_prob: p.outlier_prob,
                    outlier_prob_not: p.outlier_prob_not,
                    outlier_marginal_prob: p.outlier_marginal_prob,
                })
                .collect();

            let crp = prior
                .compute_crp_prior(&clusters, stats.num_nodes)
                .expect("crp prior should compute");
            approx_eq_tol(crp, case.expected.crp_prior, 1e-8);

            let (log_p, log_p_one) = joint
                .compute_log_p_and_log_p_one(
                    &stats,
                    &clusters,
                    Some(&likelihood),
                    &outlier_points,
                    &assigned,
                )
                .expect("joint scores should compute");

            approx_eq_tol(log_p, case.expected.log_p, 1e-8);
            approx_eq_tol(log_p_one, case.expected.log_p_one, 1e-8);

            let direct_log_p = joint
                .log_p(
                    &stats,
                    &clusters,
                    Some(&likelihood),
                    &outlier_points,
                    &assigned,
                )
                .expect("direct log_p should compute");
            let direct_log_p_one = joint
                .log_p_one(
                    &stats,
                    &clusters,
                    Some(&likelihood),
                    &outlier_points,
                    &assigned,
                )
                .expect("direct log_p_one should compute");

            approx_eq_tol(direct_log_p, case.expected.log_p, 1e-8);
            approx_eq_tol(direct_log_p_one, case.expected.log_p_one, 1e-8);

            assert!(
                log_p.is_finite() && log_p_one.is_finite(),
                "case {} produced non-finite score",
                case.name
            );
        }
    }

    // -----------------------------------------------------------------------
    // Oracle tests: verified against Python PhyClone FSCRPDistribution
    // -----------------------------------------------------------------------
    //
    // Python stores `c_const = np.log(1000)` (log-space, assigned in __init__
    // via the `c_const` setter).  Rust stores `c_const = 1000.0` (raw value)
    // and calls `.ln()` at use-time.  Both evaluate identically to:
    //
    //   log_const = ln(1000) ≈ 6.907755278982137
    //   z_term(k) = ln(1 – (1/c)^k) – ln(1 – 1/c)   (for c = 1000)
    //   r_term(k) = –(z_term(k) + log_const × (k – 1))
    //
    // Reference values below were computed from the Taylor series of ln(1–x):
    //   ln(1 – 10⁻³) ≈ –0.0010005003335835335
    //   ln(1 – 10⁻⁶) ≈ –1.0000005000001667 × 10⁻⁶
    //   ln(1 – 10⁻⁹) ≈ –1.0000000005    × 10⁻⁹

    /// Oracle: `compute_z_term` matches Python `FSCRPDistribution._compute_z_term`.
    #[test]
    fn fscrp_compute_z_term_oracle_vs_python() {
        let fscrp = CompatFscrpDistribution::default(); // alpha=1.0, c_const=1000.0

        // k=0 → early return: a_term = 0.0 × num_nodes = 0.0
        approx_eq(fscrp.compute_z_term(0, 5), 0.0);

        // k=1 → numerator == denominator → 0.0
        approx_eq(fscrp.compute_z_term(1, 3), 0.0);

        // k=2: ln(1–1e-6) – ln(1–1e-3)
        //   = –1.0000005e-6 – (–0.0010005003335835335)
        //   = 0.0009995003330835333  (Python: np.log(1-1e-6) – np.log(1-1e-3))
        approx_eq(fscrp.compute_z_term(2, 5), 9.995_003_330_835_333e-4);

        // k=3: ln(1–1e-9) – ln(1–1e-3)
        //   = –1.0e-9 – (–0.0010005003335835335)
        //   = 0.001000499333583533  (Python: np.log(1-1e-9) – np.log(1-1e-3))
        approx_eq(fscrp.compute_z_term(3, 5), 1.000_499_333_583_533e-3);
    }

    /// Oracle: `compute_r_term` matches Python `FSCRPDistribution._compute_r_term`.
    #[test]
    fn fscrp_compute_r_term_oracle_vs_python() {
        let fscrp = CompatFscrpDistribution::default();
        let log_c = 1000.0_f64.ln(); // ≈ 6.907755278982137

        // k=1: z_term=0, 0×ln(c) = 0 → r_term = 0.0
        approx_eq(fscrp.compute_r_term(1, 5), 0.0);

        // k=2: –(z_term(2) + ln(1000))
        let z2 = 9.995_003_330_835_333e-4_f64;
        approx_eq(fscrp.compute_r_term(2, 5), -(z2 + log_c));

        // k=3: –(z_term(3) + 2×ln(1000))
        let z3 = 1.000_499_333_583_533e-3_f64;
        approx_eq(fscrp.compute_r_term(3, 5), -(z3 + 2.0 * log_c));
    }

    /// Oracle: `log_p` and `log_p_one` for a 2-node star tree.
    ///
    /// Topology: root → A, root → B.  Each node has 1 data point.
    /// alpha = 1.0, c_const = 1000.0.  Likelihood = [[0, 0, 0]] (1 sample, 3 grid pts).
    ///
    /// Derived analytically:
    ///   crp_prior   = 2·ln(1) + 0 + 0 = 0.0
    ///   log_p prior = 0 – 1·ln(3) – ln(2!) = –ln(6)
    ///   tree_marginal = log_sum_exp([0,0,0]) = ln(3)
    ///   log_p joint  = –ln(6) + ln(3) = –ln(2)
    ///
    ///   r_term(k=2) = –(z_term(2) + ln(1000))
    ///   log_p_one prior = 0 + (0 + r_term) – ln(2!) = r_term – ln(2)
    ///   tree_log_likelihood_at_one = last grid point = 0.0
    ///   log_p_one joint = r_term – ln(2)
    #[test]
    fn fscrp_log_p_star_tree_oracle_vs_python() {
        let joint = CompatTreeJointDistribution {
            prior: CompatFscrpDistribution::default(),
            outlier_modelling_active: true,
        };
        let stats = CompatTreePriorStats {
            num_nodes: 2,
            multiplicity_log: 2.0_f64.ln(), // root has 2 children → log(2!)
            root_subtree_node_counts: vec![1, 1],
        };
        let clusters = vec![
            CompatNodeCluster {
                data_point_count: 1,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 1,
                is_outlier_node: false,
            },
        ];
        let likelihood = CompatTreeLikelihood {
            root_children_count: 1,
            data_log_likelihood: vec![vec![0.0, 0.0, 0.0]],
        };
        let assigned: HashSet<DataPointId> = HashSet::new();

        let (log_p, log_p_one) = joint
            .compute_log_p_and_log_p_one(&stats, &clusters, Some(&likelihood), &[], &assigned)
            .expect("oracle star tree scores should compute");

        // log_p = –ln(2)
        approx_eq(log_p, -(2.0_f64.ln()));

        // log_p_one = r_term(2) – ln(2)
        let log_c = 1000.0_f64.ln();
        let z2 = 9.995_003_330_835_333e-4_f64;
        let r_term_2 = -(z2 + log_c);
        approx_eq(log_p_one, r_term_2 - 2.0_f64.ln());
    }

    /// Oracle: `log_p` and `log_p_one` for a 2-node chain tree.
    ///
    /// Topology: root → A → B.  A has 2 data points, B has 1.
    /// alpha = 1.0, c_const = 1000.0.  Likelihood = [[0, 0, 0]] (1 sample, 3 grid pts).
    ///
    /// Derived analytically:
    ///   crp_prior   = 2·ln(1) + log(1!) + log(0!) = 0.0
    ///   multiplicity_log = 0 (every node has ≤1 child → log(1!)=0)
    ///   log_p prior = 0 – 1·ln(3) – 0 = –ln(3)
    ///   tree_marginal = ln(3)
    ///   log_p joint  = –ln(3) + ln(3) = 0.0
    ///
    ///   r_term(k=1) = 0.0  (single root subtree)
    ///   num_ways = (2–1)·ln(2) = ln(2)
    ///   log_p_one prior = 0 + (–ln(2) + 0) – 0 = –ln(2)
    ///   tree_log_likelihood_at_one = 0.0
    ///   log_p_one joint = –ln(2)
    #[test]
    fn fscrp_log_p_chain_tree_oracle_vs_python() {
        let joint = CompatTreeJointDistribution {
            prior: CompatFscrpDistribution::default(),
            outlier_modelling_active: true,
        };
        let stats = CompatTreePriorStats {
            num_nodes: 2,
            multiplicity_log: 0.0,             // every node has ≤1 child
            root_subtree_node_counts: vec![2], // A-subtree: A + B = 2 nodes
        };
        let clusters = vec![
            CompatNodeCluster {
                data_point_count: 2,
                is_outlier_node: false,
            },
            CompatNodeCluster {
                data_point_count: 1,
                is_outlier_node: false,
            },
        ];
        let likelihood = CompatTreeLikelihood {
            root_children_count: 1,
            data_log_likelihood: vec![vec![0.0, 0.0, 0.0]],
        };
        let assigned: HashSet<DataPointId> = HashSet::new();

        let (log_p, log_p_one) = joint
            .compute_log_p_and_log_p_one(&stats, &clusters, Some(&likelihood), &[], &assigned)
            .expect("oracle chain tree scores should compute");

        // log_p = 0.0
        approx_eq(log_p, 0.0);

        // log_p_one = –ln(2)
        approx_eq(log_p_one, -(2.0_f64.ln()));
    }

    #[test]
    fn make_log_p_scorer_produces_valid_closure() {
        // Verify that make_log_p_scorer creates a closure suitable for passing to proposal builders.
        use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};
        use crate::phyclone::compat::tree_model::build_star_tree;

        let joint = CompatTreeJointDistribution {
            prior: CompatFscrpDistribution {
                alpha: 1.5,
                c_const: 100.0,
            },
            outlier_modelling_active: true,
        };

        // Create minimal test data points.
        let data_points = vec![
            CompatDataPoint {
                idx: 0,
                name: CompatDataPointName::Str("dp0".to_string()),
                mutation_ids: vec!["mut1".to_string()],
                sample_ids: vec!["s1".to_string()],
                value: vec![vec![0.5, 0.8], vec![0.6, 0.7]],
                raw_outlier_prob: 0.1,
                outlier_prob: 0.1,
                outlier_prob_not: 0.9,
                outlier_marginal_prob: 0.05,
                size: 2,
            },
            CompatDataPoint {
                idx: 1,
                name: CompatDataPointName::Str("dp1".to_string()),
                mutation_ids: vec!["mut2".to_string()],
                sample_ids: vec!["s1".to_string()],
                value: vec![vec![0.3, 0.4], vec![0.35, 0.45]],
                raw_outlier_prob: 0.2,
                outlier_prob: 0.2,
                outlier_prob_not: 0.8,
                outlier_marginal_prob: 0.1,
                size: 2,
            },
        ];

        let data_points_arc = std::sync::Arc::new(data_points.clone());
        let scorer = joint.make_log_p_scorer(data_points_arc);

        // Build a simple star tree and score it.
        let node_ids: Vec<String> = (0..data_points.len())
            .map(|i| format!("node_{}", i))
            .collect();

        let dp_pairs: Vec<(usize, &str, &[Vec<f64>])> = data_points
            .iter()
            .zip(node_ids.iter())
            .map(|(dp, id)| (dp.idx, id.as_str(), dp.value.as_slice()))
            .collect();

        let tree = build_star_tree(&dp_pairs, 2, 2, 0.0).expect("star tree should build");

        let score = scorer(&tree);
        assert!(
            score.is_finite(),
            "scorer should produce finite score, got {}",
            score
        );
    }
}
