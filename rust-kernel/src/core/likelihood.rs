use statrs::function::gamma::ln_gamma;

use crate::types::{Density, SampleDataPoint};

fn log_binomial_coefficient(n: i32, x: i32) -> f64 {
    ln_gamma((n + 1) as f64) - ln_gamma((n - x + 1) as f64) - ln_gamma((x + 1) as f64)
}

fn log_pyclone_pdf(data: &SampleDataPoint, f: f64, log_pdf: impl Fn(i32, i32, f64) -> f64) -> f64 {
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
        let ll = data.log_pi[c_idx] + log_pdf(data.a + data.b, data.b, expected_vaf);
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

    let n = data.a + data.b;
    let x = data.b;
    let log_coefficient = log_binomial_coefficient(n, x);
    let beta_binomial_constant =
        log_coefficient + ln_gamma(precision) - ln_gamma(precision + n as f64);
    for (value, &ccf) in out.iter_mut().zip(ccf_grid) {
        *value = match density {
            Density::Binomial => log_pyclone_pdf(data, ccf, |_, _, p| {
                if p == 0.0 {
                    return if x == 0 { 0.0 } else { f64::NEG_INFINITY };
                }
                if p == 1.0 {
                    return if x == n { 0.0 } else { f64::NEG_INFINITY };
                }
                log_coefficient + x as f64 * p.ln() + (n - x) as f64 * (-p).ln_1p()
            }),
            Density::BetaBinomial => log_pyclone_pdf(data, ccf, |_, _, p| {
                let alpha = p * precision;
                let beta = precision - alpha;
                if alpha <= 0.0 || beta <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    beta_binomial_constant + ln_gamma(alpha + x as f64) - ln_gamma(alpha)
                        + ln_gamma(beta + (n - x) as f64)
                        - ln_gamma(beta)
                }
            }),
        };
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::compute_likelihood_grid_into;
    use crate::abi::PcvRow;
    use crate::preprocess::{build_sample_data_point, get_ccf_grid};
    use crate::types::Density;

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
        let mut ll = vec![0.0; grid.len()];
        compute_likelihood_grid_into(&data, &grid, Density::Binomial, 200.0, &mut ll).unwrap();

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
        let mut ll = vec![0.0; grid.len()];
        compute_likelihood_grid_into(&data, &grid, Density::BetaBinomial, 200.0, &mut ll).unwrap();

        assert_eq!(ll.len(), 5);
        assert!(ll.iter().all(|value| value.is_finite()));
    }
}
