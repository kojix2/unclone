use statrs::function::gamma::ln_gamma;

use crate::types::{Density, SampleDataPoint};

fn log_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

fn log_binomial_coefficient(n: i32, x: i32) -> f64 {
    ln_gamma((n + 1) as f64) - ln_gamma((n - x + 1) as f64) - ln_gamma((x + 1) as f64)
}

pub fn log_beta_binomial_pdf(n: i32, x: i32, a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return f64::NEG_INFINITY;
    }
    log_binomial_coefficient(n, x) + log_beta(a + x as f64, b + (n - x) as f64) - log_beta(a, b)
}

pub fn log_binomial_pdf(n: i32, x: i32, p: f64) -> f64 {
    if p == 0.0 {
        return if x == 0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if p == 1.0 {
        return if x == n { 0.0 } else { f64::NEG_INFINITY };
    }
    log_binomial_coefficient(n, x) + (x as f64) * p.ln() + ((n - x) as f64) * (-p).ln_1p()
}

pub fn log_pyclone_binomial_pdf(data: &SampleDataPoint, f: f64) -> f64 {
    let t = data.t;
    let population_prior = [1.0 - t, t * (1.0 - f), t * f];
    let mut max_ll = f64::NEG_INFINITY;
    let mut exp_sum = 0.0;

    for (c_idx, cn) in data.cn.iter().enumerate() {
        let mut expected_vaf = 0.0;
        let mut norm_const = 0.0;

        for i in 0..3 {
            let expected_cn = population_prior[i] * cn[i] as f64;
            expected_vaf += expected_cn * data.mu[c_idx][i];
            norm_const += expected_cn;
        }

        expected_vaf /= norm_const;
        let ll = data.log_pi[c_idx] + log_binomial_pdf(data.a + data.b, data.b, expected_vaf);
        if ll > max_ll {
            exp_sum = if max_ll.is_infinite() {
                1.0
            } else {
                exp_sum * (max_ll - ll).exp() + 1.0
            };
            max_ll = ll;
        } else {
            exp_sum += (ll - max_ll).exp();
        }
    }

    if max_ll.is_infinite() {
        max_ll
    } else {
        max_ll + exp_sum.ln()
    }
}

pub fn log_pyclone_beta_binomial_pdf(data: &SampleDataPoint, f: f64, precision: f64) -> f64 {
    let t = data.t;
    let population_prior = [1.0 - t, t * (1.0 - f), t * f];
    let mut max_ll = f64::NEG_INFINITY;
    let mut exp_sum = 0.0;

    for (c_idx, cn) in data.cn.iter().enumerate() {
        let mut expected_vaf = 0.0;
        let mut norm_const = 0.0;

        for i in 0..3 {
            let expected_cn = population_prior[i] * cn[i] as f64;
            expected_vaf += expected_cn * data.mu[c_idx][i];
            norm_const += expected_cn;
        }

        expected_vaf /= norm_const;
        let alpha = expected_vaf * precision;
        let beta = precision - alpha;
        let ll = data.log_pi[c_idx] + log_beta_binomial_pdf(data.a + data.b, data.b, alpha, beta);
        if ll > max_ll {
            exp_sum = if max_ll.is_infinite() {
                1.0
            } else {
                exp_sum * (max_ll - ll).exp() + 1.0
            };
            max_ll = ll;
        } else {
            exp_sum += (ll - max_ll).exp();
        }
    }

    if max_ll.is_infinite() {
        max_ll
    } else {
        max_ll + exp_sum.ln()
    }
}

#[allow(dead_code)]
pub fn compute_likelihood_grid(
    data: &SampleDataPoint,
    ccf_grid: &[f64],
    density: Density,
    precision: f64,
) -> Result<Vec<f64>, String> {
    if ccf_grid.is_empty() {
        return Err("ccf_grid must not be empty".to_string());
    }
    if precision <= 0.0 {
        return Err("precision must be > 0".to_string());
    }

    let grid = ccf_grid
        .iter()
        .map(|&ccf| match density {
            Density::Binomial => log_pyclone_binomial_pdf(data, ccf),
            Density::BetaBinomial => log_pyclone_beta_binomial_pdf(data, ccf, precision),
        })
        .collect();

    Ok(grid)
}

pub fn compute_likelihood_grid_into(
    data: &SampleDataPoint,
    ccf_grid: &[f64],
    density: Density,
    precision: f64,
    out: &mut [f64],
) -> Result<(), String> {
    if ccf_grid.is_empty() {
        return Err("ccf_grid must not be empty".to_string());
    }
    if precision <= 0.0 {
        return Err("precision must be > 0".to_string());
    }
    if out.len() != ccf_grid.len() {
        return Err("out length must equal ccf_grid length".to_string());
    }

    for (index, &ccf) in ccf_grid.iter().enumerate() {
        out[index] = match density {
            Density::Binomial => log_pyclone_binomial_pdf(data, ccf),
            Density::BetaBinomial => log_pyclone_beta_binomial_pdf(data, ccf, precision),
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{compute_likelihood_grid, log_beta_binomial_pdf, log_binomial_pdf};
    use crate::abi::PcvRow;
    use crate::preprocess::{build_sample_data_point, get_ccf_grid};
    use crate::types::Density;

    fn approx_eq(left: f64, right: f64, tol: f64) {
        let delta = (left - right).abs();
        assert!(
            delta < tol,
            "left={left}, right={right}, delta={delta}, tol={tol}"
        );
    }

    #[test]
    fn binomial_pdf_matches_closed_form_half_probability() {
        let actual = log_binomial_pdf(10, 5, 0.5);
        let expected = (252.0_f64 / 1024.0).ln();
        approx_eq(actual, expected, 1e-12);
    }

    #[test]
    fn beta_binomial_pdf_is_finite_for_valid_inputs() {
        let actual = log_beta_binomial_pdf(10, 5, 20.0, 20.0);
        assert!(actual.is_finite());
    }

    #[test]
    fn beta_binomial_pdf_returns_negative_infinity_for_invalid_boundary_params() {
        let actual = log_beta_binomial_pdf(10, 5, 0.0, 20.0);
        assert!(actual.is_infinite() && actual.is_sign_negative());
    }

    #[test]
    fn binomial_pdf_handles_zero_and_one_boundaries() {
        assert_eq!(log_binomial_pdf(10, 0, 0.0), 0.0);
        assert!(log_binomial_pdf(10, 1, 0.0).is_infinite());
        assert_eq!(log_binomial_pdf(10, 10, 1.0), 0.0);
        assert!(log_binomial_pdf(10, 9, 1.0).is_infinite());
    }

    #[test]
    fn computes_sample_likelihood_grid_for_binomial_density() {
        let row = PcvRow {
            mutation_index: 0,
            sample_index: 0,
            ref_counts: 10,
            alt_counts: 5,
            major_cn: 2,
            minor_cn: 1,
            normal_cn: 2,
            tumour_content: 1.0,
            error_rate: 1e-3,
        };

        let data = build_sample_data_point(&row).unwrap();
        let grid = get_ccf_grid(5, 1e-6).unwrap();
        let ll = compute_likelihood_grid(&data, &grid, Density::Binomial, 200.0).unwrap();

        assert_eq!(ll.len(), 5);
        assert!(ll.iter().all(|value| value.is_finite()));
        assert!(ll[0] < ll[1]);
        assert!(ll[1] < ll[2]);
        assert!(ll[2] < ll[3]);
        assert!(ll[3] < ll[4]);
    }

    #[test]
    fn computes_sample_likelihood_grid_for_beta_binomial_density() {
        let row = PcvRow {
            mutation_index: 0,
            sample_index: 0,
            ref_counts: 10,
            alt_counts: 5,
            major_cn: 2,
            minor_cn: 1,
            normal_cn: 2,
            tumour_content: 1.0,
            error_rate: 1e-3,
        };

        let data = build_sample_data_point(&row).unwrap();
        let grid = get_ccf_grid(5, 1e-6).unwrap();
        let ll = compute_likelihood_grid(&data, &grid, Density::BetaBinomial, 200.0).unwrap();

        assert_eq!(ll.len(), 5);
        assert!(ll.iter().all(|value| value.is_finite()));
    }
}
