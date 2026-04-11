#![allow(dead_code)]

use crate::math::log_sum_exp;

use super::data::CompatDataPoint;

pub fn compute_single_node_data_log_likelihood(
    data_points: &[CompatDataPoint],
) -> Result<f64, String> {
    let (num_samples, num_grid) = validate_point_dimensions(data_points)?;
    if num_samples == 0 || num_grid == 0 {
        return Ok(0.0);
    }

    let log_prior = -((num_grid as f64).ln());
    let mut total = 0.0;

    for sample_idx in 0..num_samples {
        let mut sample_terms = vec![0.0_f64; num_grid];

        for point in data_points {
            for (grid_idx, value) in point.value[sample_idx].iter().enumerate() {
                sample_terms[grid_idx] += *value;
            }
        }

        for term in &mut sample_terms {
            *term += log_prior;
        }

        total += log_sum_exp(&sample_terms);
    }

    Ok(total)
}

pub fn compute_chain_two_node_data_log_likelihood(
    parent_points: &[CompatDataPoint],
    child_points: &[CompatDataPoint],
) -> Result<f64, String> {
    let (num_samples, num_grid) = validate_pair_dimensions(parent_points, child_points)?;
    if num_samples == 0 || num_grid == 0 {
        return Ok(0.0);
    }

    let log_prior = -((num_grid as f64).ln());
    let mut total = 0.0;

    for sample_idx in 0..num_samples {
        let mut parent_terms = vec![log_prior; num_grid];
        for point in parent_points {
            for (grid_idx, value) in point.value[sample_idx].iter().enumerate() {
                parent_terms[grid_idx] += *value;
            }
        }

        let mut child_terms = vec![log_prior; num_grid];
        for point in child_points {
            for (grid_idx, value) in point.value[sample_idx].iter().enumerate() {
                child_terms[grid_idx] += *value;
            }
        }

        let mut child_prefix_lse = vec![f64::NEG_INFINITY; num_grid];
        let mut running = f64::NEG_INFINITY;
        for (grid_idx, child_value) in child_terms.iter().enumerate() {
            running = log_add_exp(running, *child_value);
            child_prefix_lse[grid_idx] = running;
        }

        let mut sample_terms = vec![0.0_f64; num_grid];
        for grid_idx in 0..num_grid {
            sample_terms[grid_idx] = parent_terms[grid_idx] + child_prefix_lse[grid_idx];
        }

        total += log_sum_exp(&sample_terms);
    }

    Ok(total)
}

pub fn compute_star_two_children_data_log_likelihood(
    parent_points: &[CompatDataPoint],
    child_a_points: &[CompatDataPoint],
    child_b_points: &[CompatDataPoint],
) -> Result<f64, String> {
    let (num_samples, num_grid) = validate_pair_dimensions(parent_points, child_a_points)?;
    let (check_samples, check_grid) = validate_pair_dimensions(parent_points, child_b_points)?;
    if check_samples != num_samples {
        return Err("all star branches must have the same sample count".to_string());
    }
    if check_grid != num_grid {
        return Err("all star branches must have the same grid size".to_string());
    }

    if num_samples == 0 || num_grid == 0 {
        return Ok(0.0);
    }

    let log_prior = -((num_grid as f64).ln());
    let mut total = 0.0;

    for sample_idx in 0..num_samples {
        let mut parent_terms = vec![log_prior; num_grid];
        for point in parent_points {
            for (grid_idx, value) in point.value[sample_idx].iter().enumerate() {
                parent_terms[grid_idx] += *value;
            }
        }

        let mut child_a_terms = vec![log_prior; num_grid];
        for point in child_a_points {
            for (grid_idx, value) in point.value[sample_idx].iter().enumerate() {
                child_a_terms[grid_idx] += *value;
            }
        }

        let mut child_b_terms = vec![log_prior; num_grid];
        for point in child_b_points {
            for (grid_idx, value) in point.value[sample_idx].iter().enumerate() {
                child_b_terms[grid_idx] += *value;
            }
        }

        let child_a_prefix = build_prefix_logsumexp(&child_a_terms);
        let child_b_prefix = build_prefix_logsumexp(&child_b_terms);

        let mut sample_terms = vec![0.0_f64; num_grid];
        for grid_idx in 0..num_grid {
            sample_terms[grid_idx] =
                parent_terms[grid_idx] + child_a_prefix[grid_idx] + child_b_prefix[grid_idx];
        }

        total += log_sum_exp(&sample_terms);
    }

    Ok(total)
}

pub fn compute_multi_root_two_roots_data_log_likelihood(
    root_a_points: &[CompatDataPoint],
    root_b_points: &[CompatDataPoint],
) -> Result<f64, String> {
    let (num_samples, num_grid) = validate_pair_dimensions(root_a_points, root_b_points)?;
    if num_samples == 0 || num_grid == 0 {
        return Ok(0.0);
    }

    let log_prior = -((num_grid as f64).ln());
    let mut total = 0.0;

    for sample_idx in 0..num_samples {
        let mut root_a_terms = vec![log_prior; num_grid];
        for point in root_a_points {
            for (grid_idx, value) in point.value[sample_idx].iter().enumerate() {
                root_a_terms[grid_idx] += *value;
            }
        }

        let mut root_b_terms = vec![log_prior; num_grid];
        for point in root_b_points {
            for (grid_idx, value) in point.value[sample_idx].iter().enumerate() {
                root_b_terms[grid_idx] += *value;
            }
        }

        let root_b_prefix = build_prefix_logsumexp(&root_b_terms);

        let mut sample_terms = vec![0.0_f64; num_grid];
        for root_a_idx in 0..num_grid {
            let max_root_b_idx = (num_grid - 1) - root_a_idx;
            sample_terms[root_a_idx] = root_a_terms[root_a_idx] + root_b_prefix[max_root_b_idx];
        }

        total += log_sum_exp(&sample_terms);
    }

    Ok(total)
}

fn validate_point_dimensions(data_points: &[CompatDataPoint]) -> Result<(usize, usize), String> {
    let Some(first) = data_points.first() else {
        return Ok((0, 0));
    };

    let num_samples = first.value.len();
    let num_grid = first.value.first().map_or(0, Vec::len);

    for point in data_points {
        if point.value.len() != num_samples {
            return Err("all datapoints must have the same sample count".to_string());
        }
        for sample in &point.value {
            if sample.len() != num_grid {
                return Err("all datapoints must have the same grid size".to_string());
            }
        }
    }

    Ok((num_samples, num_grid))
}

fn validate_pair_dimensions(
    parent_points: &[CompatDataPoint],
    child_points: &[CompatDataPoint],
) -> Result<(usize, usize), String> {
    let (parent_samples, parent_grid) = validate_point_dimensions(parent_points)?;
    let (child_samples, child_grid) = validate_point_dimensions(child_points)?;

    if parent_samples != child_samples {
        return Err("parent/child datapoints must have the same sample count".to_string());
    }
    if parent_grid != child_grid {
        return Err("parent/child datapoints must have the same grid size".to_string());
    }

    Ok((parent_samples, parent_grid))
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }

    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

fn build_prefix_logsumexp(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NEG_INFINITY; values.len()];
    let mut running = f64::NEG_INFINITY;

    for (idx, value) in values.iter().enumerate() {
        running = log_add_exp(running, *value);
        out[idx] = running;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::{
        compute_chain_two_node_data_log_likelihood,
        compute_multi_root_two_roots_data_log_likelihood, compute_single_node_data_log_likelihood,
        compute_star_two_children_data_log_likelihood,
    };
    use crate::phyclone::compat::data::{CompatDataPoint, CompatDataPointName};

    fn make_point(idx: usize, value: Vec<Vec<f64>>) -> CompatDataPoint {
        CompatDataPoint {
            idx,
            name: CompatDataPointName::Str(format!("dp-{}", idx)),
            mutation_ids: vec![format!("mut-{}", idx)],
            sample_ids: Vec::new(),
            value,
            raw_outlier_prob: 0.0,
            outlier_prob: 0.0,
            outlier_prob_not: 0.0,
            outlier_marginal_prob: 0.0,
            size: 1,
        }
    }

    fn lse(values: &[f64]) -> f64 {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max.is_infinite() && max.is_sign_negative() {
            return max;
        }
        let sum: f64 = values.iter().map(|v| (*v - max).exp()).sum();
        max + sum.ln()
    }

    #[test]
    fn single_node_likelihood_matches_manual_sum_for_multiple_samples() {
        let points = vec![
            make_point(0, vec![vec![-2.0, -1.0, -3.0], vec![-1.5, -0.2, -2.4]]),
            make_point(1, vec![vec![-0.5, -1.2, -0.7], vec![-1.1, -0.9, -0.8]]),
        ];

        let log_prior = -((3_f64).ln());
        let expected_sample0 = lse(&[
            -2.0 - 0.5 + log_prior,
            -1.0 - 1.2 + log_prior,
            -3.0 - 0.7 + log_prior,
        ]);
        let expected_sample1 = lse(&[
            -1.5 - 1.1 + log_prior,
            -0.2 - 0.9 + log_prior,
            -2.4 - 0.8 + log_prior,
        ]);
        let expected = expected_sample0 + expected_sample1;

        let actual = compute_single_node_data_log_likelihood(&points)
            .expect("single-node likelihood should be computed");

        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn errors_on_sample_count_mismatch() {
        let points = vec![
            make_point(0, vec![vec![-1.0, -2.0]]),
            make_point(1, vec![vec![-1.0, -2.0], vec![-0.1, -0.2]]),
        ];

        let err = compute_single_node_data_log_likelihood(&points)
            .expect_err("dimension mismatch must error");
        assert!(err.contains("sample count"));
    }

    #[test]
    fn errors_on_grid_size_mismatch() {
        let points = vec![
            make_point(0, vec![vec![-1.0, -2.0]]),
            make_point(1, vec![vec![-1.0, -2.0, -3.0]]),
        ];

        let err = compute_single_node_data_log_likelihood(&points)
            .expect_err("dimension mismatch must error");
        assert!(err.contains("grid size"));
    }

    #[test]
    fn chain_two_node_likelihood_matches_bruteforce_for_single_sample() {
        let parent_points = vec![make_point(0, vec![vec![-0.2, -1.1, -0.7]])];
        let child_points = vec![make_point(1, vec![vec![-1.2, -0.3, -2.0]])];

        let log_prior = -((3_f64).ln());
        let mut brute_terms = Vec::new();
        for parent_idx in 0..3 {
            for child_idx in 0..=parent_idx {
                brute_terms.push(
                    parent_points[0].value[0][parent_idx]
                        + child_points[0].value[0][child_idx]
                        + log_prior
                        + log_prior,
                );
            }
        }
        let expected = lse(&brute_terms);

        let actual = compute_chain_two_node_data_log_likelihood(&parent_points, &child_points)
            .expect("chain likelihood should be computed");
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn chain_two_node_likelihood_matches_bruteforce_for_multiple_samples_and_points() {
        let parent_points = vec![
            make_point(0, vec![vec![-0.2, -1.1, -0.7], vec![-0.5, -0.4, -1.9]]),
            make_point(1, vec![vec![-1.3, -0.2, -0.8], vec![-0.1, -1.6, -0.7]]),
        ];
        let child_points = vec![
            make_point(2, vec![vec![-1.2, -0.3, -2.0], vec![-0.9, -0.2, -1.1]]),
            make_point(3, vec![vec![-0.7, -0.5, -1.4], vec![-1.3, -0.8, -0.4]]),
        ];

        let log_prior = -((3_f64).ln());
        let mut expected = 0.0;
        for sample_idx in 0..2 {
            let mut brute_terms = Vec::new();
            for parent_idx in 0..3 {
                for child_idx in 0..=parent_idx {
                    let parent_sum: f64 = parent_points
                        .iter()
                        .map(|p| p.value[sample_idx][parent_idx])
                        .sum();
                    let child_sum: f64 = child_points
                        .iter()
                        .map(|p| p.value[sample_idx][child_idx])
                        .sum();
                    brute_terms.push(parent_sum + child_sum + log_prior + log_prior);
                }
            }
            expected += lse(&brute_terms);
        }

        let actual = compute_chain_two_node_data_log_likelihood(&parent_points, &child_points)
            .expect("chain likelihood should be computed");
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn chain_two_node_errors_on_parent_child_sample_mismatch() {
        let parent_points = vec![make_point(0, vec![vec![-1.0, -2.0]])];
        let child_points = vec![make_point(1, vec![vec![-1.0, -2.0], vec![-0.1, -0.2]])];

        let err = compute_chain_two_node_data_log_likelihood(&parent_points, &child_points)
            .expect_err("sample mismatch must error");
        assert!(err.contains("sample count"));
    }

    #[test]
    fn chain_two_node_errors_on_parent_child_grid_mismatch() {
        let parent_points = vec![make_point(0, vec![vec![-1.0, -2.0]])];
        let child_points = vec![make_point(1, vec![vec![-1.0, -2.0, -3.0]])];

        let err = compute_chain_two_node_data_log_likelihood(&parent_points, &child_points)
            .expect_err("grid mismatch must error");
        assert!(err.contains("grid size"));
    }

    #[test]
    fn star_two_children_likelihood_matches_bruteforce_for_single_sample() {
        let parent_points = vec![make_point(0, vec![vec![-0.2, -1.1, -0.7]])];
        let child_a_points = vec![make_point(1, vec![vec![-1.2, -0.3, -2.0]])];
        let child_b_points = vec![make_point(2, vec![vec![-0.6, -0.7, -1.5]])];

        let log_prior = -((3_f64).ln());
        let mut brute_terms = Vec::new();
        for parent_idx in 0..3 {
            for child_a_idx in 0..=parent_idx {
                for child_b_idx in 0..=parent_idx {
                    brute_terms.push(
                        parent_points[0].value[0][parent_idx]
                            + child_a_points[0].value[0][child_a_idx]
                            + child_b_points[0].value[0][child_b_idx]
                            + log_prior
                            + log_prior
                            + log_prior,
                    );
                }
            }
        }
        let expected = lse(&brute_terms);

        let actual = compute_star_two_children_data_log_likelihood(
            &parent_points,
            &child_a_points,
            &child_b_points,
        )
        .expect("star likelihood should be computed");
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn star_two_children_likelihood_matches_bruteforce_for_multiple_samples_and_points() {
        let parent_points = vec![
            make_point(0, vec![vec![-0.2, -1.1, -0.7], vec![-0.5, -0.4, -1.9]]),
            make_point(1, vec![vec![-1.3, -0.2, -0.8], vec![-0.1, -1.6, -0.7]]),
        ];
        let child_a_points = vec![
            make_point(2, vec![vec![-1.2, -0.3, -2.0], vec![-0.9, -0.2, -1.1]]),
            make_point(3, vec![vec![-0.7, -0.5, -1.4], vec![-1.3, -0.8, -0.4]]),
        ];
        let child_b_points = vec![
            make_point(4, vec![vec![-0.6, -1.4, -0.9], vec![-0.3, -1.2, -0.5]]),
            make_point(5, vec![vec![-0.8, -0.9, -1.1], vec![-1.0, -0.6, -0.7]]),
        ];

        let log_prior = -((3_f64).ln());
        let mut expected = 0.0;
        for sample_idx in 0..2 {
            let mut brute_terms = Vec::new();
            for parent_idx in 0..3 {
                for child_a_idx in 0..=parent_idx {
                    for child_b_idx in 0..=parent_idx {
                        let parent_sum: f64 = parent_points
                            .iter()
                            .map(|p| p.value[sample_idx][parent_idx])
                            .sum();
                        let child_a_sum: f64 = child_a_points
                            .iter()
                            .map(|p| p.value[sample_idx][child_a_idx])
                            .sum();
                        let child_b_sum: f64 = child_b_points
                            .iter()
                            .map(|p| p.value[sample_idx][child_b_idx])
                            .sum();
                        brute_terms.push(
                            parent_sum
                                + child_a_sum
                                + child_b_sum
                                + log_prior
                                + log_prior
                                + log_prior,
                        );
                    }
                }
            }
            expected += lse(&brute_terms);
        }

        let actual = compute_star_two_children_data_log_likelihood(
            &parent_points,
            &child_a_points,
            &child_b_points,
        )
        .expect("star likelihood should be computed");
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn star_two_children_errors_on_branch_sample_mismatch() {
        let parent_points = vec![make_point(0, vec![vec![-1.0, -2.0]])];
        let child_a_points = vec![make_point(1, vec![vec![-1.0, -2.0]])];
        let child_b_points = vec![make_point(2, vec![vec![-1.0, -2.0], vec![-0.1, -0.2]])];

        let err = compute_star_two_children_data_log_likelihood(
            &parent_points,
            &child_a_points,
            &child_b_points,
        )
        .expect_err("sample mismatch must error");
        assert!(err.contains("sample count"));
    }

    #[test]
    fn star_two_children_errors_on_branch_grid_mismatch() {
        let parent_points = vec![make_point(0, vec![vec![-1.0, -2.0]])];
        let child_a_points = vec![make_point(1, vec![vec![-1.0, -2.0]])];
        let child_b_points = vec![make_point(2, vec![vec![-1.0, -2.0, -3.0]])];

        let err = compute_star_two_children_data_log_likelihood(
            &parent_points,
            &child_a_points,
            &child_b_points,
        )
        .expect_err("grid mismatch must error");
        assert!(err.contains("grid size"));
    }

    #[test]
    fn multi_root_two_roots_likelihood_matches_bruteforce_for_single_sample() {
        let root_a_points = vec![make_point(0, vec![vec![-0.2, -1.1, -0.7]])];
        let root_b_points = vec![make_point(1, vec![vec![-1.2, -0.3, -2.0]])];

        let log_prior = -((3_f64).ln());
        let mut brute_terms = Vec::new();
        for root_a_idx in 0..3 {
            for root_b_idx in 0..3 {
                if root_a_idx + root_b_idx > 2 {
                    continue;
                }
                brute_terms.push(
                    root_a_points[0].value[0][root_a_idx]
                        + root_b_points[0].value[0][root_b_idx]
                        + log_prior
                        + log_prior,
                );
            }
        }
        let expected = lse(&brute_terms);

        let actual =
            compute_multi_root_two_roots_data_log_likelihood(&root_a_points, &root_b_points)
                .expect("multi-root likelihood should be computed");
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn multi_root_two_roots_likelihood_matches_bruteforce_for_multiple_samples_and_points() {
        let root_a_points = vec![
            make_point(0, vec![vec![-0.2, -1.1, -0.7], vec![-0.5, -0.4, -1.9]]),
            make_point(1, vec![vec![-1.3, -0.2, -0.8], vec![-0.1, -1.6, -0.7]]),
        ];
        let root_b_points = vec![
            make_point(2, vec![vec![-1.2, -0.3, -2.0], vec![-0.9, -0.2, -1.1]]),
            make_point(3, vec![vec![-0.7, -0.5, -1.4], vec![-1.3, -0.8, -0.4]]),
        ];

        let log_prior = -((3_f64).ln());
        let mut expected = 0.0;
        for sample_idx in 0..2 {
            let mut brute_terms = Vec::new();
            for root_a_idx in 0..3 {
                for root_b_idx in 0..3 {
                    if root_a_idx + root_b_idx > 2 {
                        continue;
                    }
                    let root_a_sum: f64 = root_a_points
                        .iter()
                        .map(|p| p.value[sample_idx][root_a_idx])
                        .sum();
                    let root_b_sum: f64 = root_b_points
                        .iter()
                        .map(|p| p.value[sample_idx][root_b_idx])
                        .sum();
                    brute_terms.push(root_a_sum + root_b_sum + log_prior + log_prior);
                }
            }
            expected += lse(&brute_terms);
        }

        let actual =
            compute_multi_root_two_roots_data_log_likelihood(&root_a_points, &root_b_points)
                .expect("multi-root likelihood should be computed");
        assert!((actual - expected).abs() <= 1e-10);
    }

    #[test]
    fn multi_root_two_roots_errors_on_sample_mismatch() {
        let root_a_points = vec![make_point(0, vec![vec![-1.0, -2.0]])];
        let root_b_points = vec![make_point(1, vec![vec![-1.0, -2.0], vec![-0.1, -0.2]])];

        let err = compute_multi_root_two_roots_data_log_likelihood(&root_a_points, &root_b_points)
            .expect_err("sample mismatch must error");
        assert!(err.contains("sample count"));
    }

    #[test]
    fn multi_root_two_roots_errors_on_grid_mismatch() {
        let root_a_points = vec![make_point(0, vec![vec![-1.0, -2.0]])];
        let root_b_points = vec![make_point(1, vec![vec![-1.0, -2.0, -3.0]])];

        let err = compute_multi_root_two_roots_data_log_likelihood(&root_a_points, &root_b_points)
            .expect_err("grid mismatch must error");
        assert!(err.contains("grid size"));
    }
}
