use crate::abi::PcvRow;
use crate::likelihood::compute_likelihood_grid_into;
use crate::types::{Density, LogLikelihoodTensor, MajorCnPrior, SampleDataPoint};
use rayon::prelude::*;

pub fn get_major_cn_prior(
    major_cn: i32,
    minor_cn: i32,
    normal_cn: i32,
    error_rate: f64,
) -> Result<MajorCnPrior, String> {
    if major_cn <= 0 {
        return Err("major_cn must be > 0".to_string());
    }
    if minor_cn < 0 || normal_cn <= 0 {
        return Err("minor_cn must be >= 0 and normal_cn must be > 0".to_string());
    }
    if major_cn < minor_cn {
        return Err("major_cn must be >= minor_cn".to_string());
    }
    if !(0.0..1.0).contains(&error_rate) {
        return Err("error_rate must be in [0, 1)".to_string());
    }

    let total_cn = major_cn + minor_cn;
    let mut cn = Vec::with_capacity((major_cn + 1) as usize);
    let mut mu = Vec::with_capacity((major_cn + 1) as usize);

    for x in 1..=major_cn {
        cn.push([normal_cn, normal_cn, total_cn]);
        mu.push([
            error_rate,
            error_rate,
            (x as f64 / total_cn as f64).min(1.0 - error_rate),
        ]);
    }

    if total_cn != normal_cn {
        let mutation_after_cn = [normal_cn, total_cn, total_cn];
        if !cn.contains(&mutation_after_cn) {
            cn.push(mutation_after_cn);
            mu.push([
                error_rate,
                error_rate,
                (1.0 / total_cn as f64).min(1.0 - error_rate),
            ]);
        }
    }

    let log_pi_val = -((cn.len() as f64).ln());
    let log_pi = vec![log_pi_val; cn.len()];

    Ok(MajorCnPrior { cn, mu, log_pi })
}

pub fn get_ccf_grid(grid_size: usize, eps: f64) -> Result<Vec<f64>, String> {
    if grid_size < 2 {
        return Err("grid_size must be >= 2".to_string());
    }
    if !(0.0..0.5).contains(&eps) {
        return Err("eps must be in (0, 0.5)".to_string());
    }

    let step = (1.0 - (2.0 * eps)) / ((grid_size - 1) as f64);
    let mut grid = Vec::with_capacity(grid_size);
    for idx in 0..grid_size {
        grid.push(eps + step * idx as f64);
    }
    Ok(grid)
}

pub fn build_sample_data_point(row: &PcvRow) -> Result<SampleDataPoint, String> {
    if row.ref_counts < 0 || row.alt_counts < 0 {
        return Err("ref_counts and alt_counts must be >= 0".to_string());
    }
    if !(0.0..=1.0).contains(&row.tumour_content) {
        return Err("tumour_content must be in [0, 1]".to_string());
    }

    let prior = get_major_cn_prior(row.major_cn, row.minor_cn, row.normal_cn, row.error_rate)?;

    Ok(SampleDataPoint {
        a: row.ref_counts,
        b: row.alt_counts,
        cn: prior.cn,
        mu: prior.mu,
        log_pi: prior.log_pi,
        t: row.tumour_content,
    })
}

pub fn build_log_p_data(
    rows: &[PcvRow],
    num_mutations: usize,
    num_samples: usize,
    ccf_grid: &[f64],
    density: Density,
    precision: f64,
) -> Result<LogLikelihoodTensor, String> {
    if rows.is_empty() {
        return Err("rows must not be empty".to_string());
    }
    if num_mutations == 0 || num_samples == 0 {
        return Err("num_mutations and num_samples must be > 0".to_string());
    }
    let num_grid_points = ccf_grid.len();
    let mut values = vec![0.0; num_mutations * num_samples * num_grid_points];
    let ordered_rows = validate_and_order_rows(rows, num_mutations, num_samples)?;

    for (pair_offset, row) in ordered_rows.iter().enumerate() {
        let Some(row) = row else {
            continue;
        };
        let data = build_sample_data_point(row)?;
        let tensor_offset = pair_offset * num_grid_points;
        compute_likelihood_grid_into(
            &data,
            ccf_grid,
            density,
            precision,
            &mut values[tensor_offset..tensor_offset + num_grid_points],
        )?;
    }

    Ok(LogLikelihoodTensor {
        num_mutations,
        num_samples,
        num_grid_points,
        values,
    })
}

pub fn build_log_p_data_parallel(
    rows: &[PcvRow],
    num_mutations: usize,
    num_samples: usize,
    ccf_grid: &[f64],
    density: Density,
    precision: f64,
) -> Result<LogLikelihoodTensor, String> {
    if rows.is_empty() {
        return Err("rows must not be empty".to_string());
    }
    if num_mutations == 0 || num_samples == 0 {
        return Err("num_mutations and num_samples must be > 0".to_string());
    }
    let num_grid_points = ccf_grid.len();
    let ordered_rows = validate_and_order_rows(rows, num_mutations, num_samples)?;
    let mut values = vec![0.0; num_mutations * num_samples * num_grid_points];
    values
        .par_chunks_mut(num_grid_points)
        .enumerate()
        .try_for_each(|(pair_offset, chunk)| -> Result<(), String> {
            let Some(row) = ordered_rows[pair_offset] else {
                return Ok(());
            };
            let data = build_sample_data_point(row)?;
            compute_likelihood_grid_into(&data, ccf_grid, density, precision, chunk)
        })?;

    Ok(LogLikelihoodTensor {
        num_mutations,
        num_samples,
        num_grid_points,
        values,
    })
}

fn validate_and_order_rows(
    rows: &[PcvRow],
    num_mutations: usize,
    num_samples: usize,
) -> Result<Vec<Option<&PcvRow>>, String> {
    let mut seen = vec![false; num_mutations * num_samples];
    let mut ordered_rows = vec![None; num_mutations * num_samples];

    for row in rows {
        if row.mutation_index < 0 || row.sample_index < 0 {
            return Err("mutation_index and sample_index must be >= 0".to_string());
        }

        let mutation_index = row.mutation_index as usize;
        let sample_index = row.sample_index as usize;

        if mutation_index >= num_mutations || sample_index >= num_samples {
            return Err("row index out of bounds for tensor shape".to_string());
        }

        let pair_offset = mutation_index * num_samples + sample_index;
        if seen[pair_offset] {
            return Err("duplicate mutation/sample pair encountered".to_string());
        }
        seen[pair_offset] = true;

        ordered_rows[pair_offset] = Some(row);
    }

    Ok(ordered_rows)
}

#[cfg(test)]
mod tests {
    use super::{
        build_log_p_data, build_log_p_data_parallel, build_sample_data_point, get_ccf_grid,
        get_major_cn_prior,
    };
    use crate::abi::PcvRow;
    use crate::types::Density;

    fn approx_eq(left: f64, right: f64) {
        let delta = (left - right).abs();
        assert!(delta < 1e-12, "left={left}, right={right}, delta={delta}");
    }

    #[test]
    fn major_cn_prior_matches_python_shape_and_values() {
        let prior = get_major_cn_prior(2, 1, 2, 1e-3).unwrap();

        assert_eq!(prior.cn, vec![[2, 2, 3], [2, 2, 3], [2, 3, 3]]);
        approx_eq(prior.mu[0][2], 1.0 / 3.0);
        approx_eq(prior.mu[1][2], 2.0 / 3.0);
        approx_eq(prior.mu[2][2], 1.0 / 3.0);
        assert_eq!(prior.log_pi.len(), 3);
        approx_eq(prior.log_pi[0], -3.0_f64.ln());
    }

    #[test]
    fn major_cn_prior_avoids_duplicate_after_cn_state() {
        let prior = get_major_cn_prior(2, 0, 2, 1e-3).unwrap();

        assert_eq!(prior.cn, vec![[2, 2, 2], [2, 2, 2]]);
        assert_eq!(prior.mu.len(), 2);
        assert_eq!(prior.log_pi.len(), 2);
    }

    #[test]
    fn major_cn_prior_rejects_major_smaller_than_minor() {
        let error = get_major_cn_prior(1, 2, 2, 1e-3).unwrap_err();

        assert_eq!(error, "major_cn must be >= minor_cn");
    }

    #[test]
    fn ccf_grid_matches_numpy_linspace_endpoints() {
        let grid = get_ccf_grid(5, 1e-6).unwrap();

        assert_eq!(grid.len(), 5);
        approx_eq(grid[0], 1e-6);
        approx_eq(grid[4], 1.0 - 1e-6);
        approx_eq(grid[2], 0.5);
    }

    #[test]
    fn builds_sample_data_point_from_row() {
        let row = PcvRow {
            mutation_index: 0,
            sample_index: 0,
            ref_counts: 10,
            alt_counts: 5,
            major_cn: 2,
            minor_cn: 1,
            normal_cn: 2,
            tumour_content: 0.8,
            error_rate: 1e-3,
        };

        let data = build_sample_data_point(&row).unwrap();
        assert_eq!(data.a, 10);
        assert_eq!(data.b, 5);
        assert_eq!(data.cn.len(), 3);
        approx_eq(data.t, 0.8);
    }

    #[test]
    fn builds_full_log_p_data_tensor() {
        let rows = vec![
            PcvRow {
                mutation_index: 0,
                sample_index: 0,
                ref_counts: 10,
                alt_counts: 5,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 1.0,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 0,
                sample_index: 1,
                ref_counts: 12,
                alt_counts: 4,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.9,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 1,
                sample_index: 0,
                ref_counts: 9,
                alt_counts: 7,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.8,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 1,
                sample_index: 1,
                ref_counts: 14,
                alt_counts: 3,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.7,
                error_rate: 1e-3,
            },
        ];

        let grid = get_ccf_grid(5, 1e-6).unwrap();
        let tensor = build_log_p_data(&rows, 2, 2, &grid, Density::Binomial, 200.0).unwrap();

        assert_eq!(tensor.num_mutations, 2);
        assert_eq!(tensor.num_samples, 2);
        assert_eq!(tensor.num_grid_points, 5);
        assert_eq!(tensor.values.len(), 20);
        assert!(tensor.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_duplicate_mutation_sample_pairs() {
        let rows = vec![
            PcvRow {
                mutation_index: 0,
                sample_index: 0,
                ref_counts: 10,
                alt_counts: 5,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 1.0,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 0,
                sample_index: 0,
                ref_counts: 11,
                alt_counts: 4,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 1.0,
                error_rate: 1e-3,
            },
        ];

        let grid = get_ccf_grid(5, 1e-6).unwrap();
        let error = build_log_p_data(&rows, 1, 2, &grid, Density::Binomial, 200.0).unwrap_err();

        assert_eq!(error, "duplicate mutation/sample pair encountered");
    }

    #[test]
    fn parallel_log_p_data_matches_sequential() {
        let rows = vec![
            PcvRow {
                mutation_index: 0,
                sample_index: 0,
                ref_counts: 10,
                alt_counts: 5,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 1.0,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 0,
                sample_index: 1,
                ref_counts: 12,
                alt_counts: 4,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.9,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 1,
                sample_index: 0,
                ref_counts: 9,
                alt_counts: 7,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.8,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 1,
                sample_index: 1,
                ref_counts: 14,
                alt_counts: 3,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.7,
                error_rate: 1e-3,
            },
        ];

        let grid = get_ccf_grid(5, 1e-6).unwrap();
        let sequential =
            build_log_p_data(&rows, 2, 2, &grid, Density::BetaBinomial, 200.0).unwrap();
        let parallel =
            build_log_p_data_parallel(&rows, 2, 2, &grid, Density::BetaBinomial, 200.0).unwrap();

        assert_eq!(sequential, parallel);
    }

    #[test]
    fn treats_missing_mutation_sample_pairs_as_neutral_likelihoods() {
        let rows = vec![
            PcvRow {
                mutation_index: 0,
                sample_index: 0,
                ref_counts: 10,
                alt_counts: 5,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 1.0,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 1,
                sample_index: 0,
                ref_counts: 11,
                alt_counts: 4,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 1.0,
                error_rate: 1e-3,
            },
        ];

        let grid = get_ccf_grid(5, 1e-6).unwrap();
        let tensor = build_log_p_data(&rows, 2, 2, &grid, Density::Binomial, 200.0).unwrap();

        assert_eq!(tensor.values.len(), 20);
        let missing_pair_offset = 1;
        let missing_values = &tensor.values
            [missing_pair_offset * grid.len()..(missing_pair_offset + 1) * grid.len()];
        assert!(missing_values.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn zero_depth_rows_have_neutral_likelihoods() {
        let rows = vec![PcvRow {
            mutation_index: 0,
            sample_index: 0,
            ref_counts: 0,
            alt_counts: 0,
            major_cn: 2,
            minor_cn: 1,
            normal_cn: 2,
            tumour_content: 1.0,
            error_rate: 1e-3,
        }];

        let grid = get_ccf_grid(5, 1e-6).unwrap();
        let tensor = build_log_p_data(&rows, 1, 1, &grid, Density::BetaBinomial, 200.0).unwrap();

        assert!(tensor.values.iter().all(|value| value.abs() < 1e-12));
    }
}
