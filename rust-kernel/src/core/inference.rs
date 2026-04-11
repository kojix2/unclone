use ndarray::{ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, Axis};
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use rayon::prelude::*;
use statrs::function::gamma::{digamma, ln_gamma};
use std::time::{Duration, Instant};

use crate::math::log_sum_exp;
use crate::types::{DataPreprocessor, LogLikelihoodTensor, Priors, VariationalParameters};

impl DataPreprocessor {
    pub fn new(log_p_data: &LogLikelihoodTensor, use_parallel: bool) -> Self {
        let theta_update_data = log_p_data.values.clone();
        let z_update_data = ArrayView3::from_shape(
            (
                log_p_data.num_mutations,
                log_p_data.num_samples,
                log_p_data.num_grid_points,
            ),
            &log_p_data.values,
        )
        .expect("log likelihood tensor shape must match backing storage")
        .permuted_axes([0, 2, 1])
        .iter()
        .copied()
        .collect();

        Self {
            theta_update_data,
            z_update_data,
            theta_update_shape: (log_p_data.num_samples, log_p_data.num_grid_points),
            z_update_shape: log_p_data.num_mutations,
            use_parallel,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct FitProfile {
    pub initial_elbo: Duration,
    pub update_z: Duration,
    pub update_pi: Duration,
    pub update_theta: Duration,
    pub iter_elbo: Duration,
    pub iterations: usize,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct FitDetailProfile {
    pub update_z_contract: Duration,
    pub update_z_normalize: Duration,
    pub update_theta_contract: Duration,
    pub update_theta_normalize: Duration,
    pub elbo_e_log_p: Duration,
    pub elbo_e_log_q: Duration,
    pub e_log_p_z_sums: Duration,
    pub e_log_p_pi_term: Duration,
    pub e_log_p_theta_prior: Duration,
    pub e_log_p_data_theta: Duration,
    pub e_log_p_data_accum: Duration,
    pub e_log_q_pi_term: Duration,
    pub e_log_q_theta_term: Duration,
    pub e_log_q_z_term: Duration,
}

impl FitProfile {
    fn print_summary(&self) {
        eprintln!(
            "[tyclone-profile] iterations={} initial_elbo_ms={:.3} update_z_ms={:.3} update_pi_ms={:.3} update_theta_ms={:.3} iter_elbo_ms={:.3}",
            self.iterations,
            self.initial_elbo.as_secs_f64() * 1_000.0,
            self.update_z.as_secs_f64() * 1_000.0,
            self.update_pi.as_secs_f64() * 1_000.0,
            self.update_theta.as_secs_f64() * 1_000.0,
            self.iter_elbo.as_secs_f64() * 1_000.0,
        );
    }
}

impl FitDetailProfile {
    fn print_summary(&self) {
        eprintln!(
            "[tyclone-fit-detail] update_z_contract_ms={:.3} update_z_normalize_ms={:.3} update_theta_contract_ms={:.3} update_theta_normalize_ms={:.3} elbo_e_log_p_ms={:.3} elbo_e_log_q_ms={:.3} e_log_p_z_sums_ms={:.3} e_log_p_pi_term_ms={:.3} e_log_p_theta_prior_ms={:.3} e_log_p_data_theta_ms={:.3} e_log_p_data_accum_ms={:.3} e_log_q_pi_term_ms={:.3} e_log_q_theta_term_ms={:.3} e_log_q_z_term_ms={:.3}",
            self.update_z_contract.as_secs_f64() * 1_000.0,
            self.update_z_normalize.as_secs_f64() * 1_000.0,
            self.update_theta_contract.as_secs_f64() * 1_000.0,
            self.update_theta_normalize.as_secs_f64() * 1_000.0,
            self.elbo_e_log_p.as_secs_f64() * 1_000.0,
            self.elbo_e_log_q.as_secs_f64() * 1_000.0,
            self.e_log_p_z_sums.as_secs_f64() * 1_000.0,
            self.e_log_p_pi_term.as_secs_f64() * 1_000.0,
            self.e_log_p_theta_prior.as_secs_f64() * 1_000.0,
            self.e_log_p_data_theta.as_secs_f64() * 1_000.0,
            self.e_log_p_data_accum.as_secs_f64() * 1_000.0,
            self.e_log_q_pi_term.as_secs_f64() * 1_000.0,
            self.e_log_q_theta_term.as_secs_f64() * 1_000.0,
            self.e_log_q_z_term.as_secs_f64() * 1_000.0,
        );
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InferenceScratch {
    reshaped_theta: Vec<f64>,
    transposed_z: Vec<f64>,
    z_sums: Vec<f64>,
    cached_e_log_p_data_theta: Option<f64>,
    log_p_data_theta: Vec<f64>,
    log_p_data_z: Vec<f64>,
}

impl InferenceScratch {
    fn new(var_params: &VariationalParameters, data_preproc: &DataPreprocessor) -> Self {
        let contraction_axis_size = var_params.num_dims * var_params.num_grid_points;

        let mut scratch = Self {
            reshaped_theta: vec![0.0; contraction_axis_size * var_params.num_clusters],
            transposed_z: vec![0.0; var_params.num_data_points * var_params.num_clusters],
            z_sums: vec![0.0; var_params.num_clusters],
            cached_e_log_p_data_theta: None,
            log_p_data_theta: vec![0.0; data_preproc.z_update_shape * var_params.num_clusters],
            log_p_data_z: vec![
                0.0;
                var_params.num_clusters
                    * var_params.num_dims
                    * var_params.num_grid_points
            ],
        };
        refresh_z_views(var_params, &mut scratch);
        scratch
    }
}

fn refresh_z_views(var_params: &VariationalParameters, scratch: &mut InferenceScratch) {
    scratch.cached_e_log_p_data_theta = None;
    let z_view = ArrayView2::from_shape(
        (var_params.num_data_points, var_params.num_clusters),
        &var_params.z,
    )
    .expect("variational z shape must match backing storage");

    for (dst, src) in scratch
        .transposed_z
        .iter_mut()
        .zip(z_view.t().iter().copied())
    {
        *dst = src;
    }

    for (dst, src) in scratch
        .z_sums
        .iter_mut()
        .zip(z_view.sum_axis(Axis(0)).iter())
    {
        *dst = *src;
    }
}

fn reshape_theta_for_z(
    theta: &[f64],
    var_params: &VariationalParameters,
    reshaped_theta: &mut [f64],
) {
    let theta_view = ArrayView3::from_shape(
        (
            var_params.num_clusters,
            var_params.num_dims,
            var_params.num_grid_points,
        ),
        theta,
    )
    .expect("theta shape must match backing storage");

    for (dst, src) in reshaped_theta
        .iter_mut()
        .zip(theta_view.permuted_axes([2, 1, 0]).iter().copied())
    {
        *dst = src;
    }
}

fn fill_log_p_data_theta<'a>(
    theta: &[f64],
    var_params: &VariationalParameters,
    data_preproc: &DataPreprocessor,
    scratch: &'a mut InferenceScratch,
) -> &'a mut [f64] {
    let contraction_axis_size = var_params.num_dims * var_params.num_grid_points;
    reshape_theta_for_z(theta, var_params, &mut scratch.reshaped_theta);

    let reshaped_theta = ArrayView2::from_shape(
        (contraction_axis_size, var_params.num_clusters),
        &scratch.reshaped_theta,
    )
    .expect("reshaped theta view must match backing storage");
    let result = &mut scratch.log_p_data_theta;
    if data_preproc.use_parallel {
        result
            .par_chunks_mut(var_params.num_clusters)
            .enumerate()
            .for_each(|(mutation_index, result_row)| {
                let data_row = ArrayView1::from(
                    &data_preproc.z_update_data[mutation_index * contraction_axis_size
                        ..(mutation_index + 1) * contraction_axis_size],
                );
                result_row.fill(0.0);
                for (contraction_index, data_value) in data_row.iter().enumerate() {
                    let theta_row = reshaped_theta.row(contraction_index);
                    for cluster_index in 0..var_params.num_clusters {
                        result_row[cluster_index] += *data_value * theta_row[cluster_index];
                    }
                }
            });
    } else {
        for mutation_index in 0..data_preproc.z_update_shape {
            let result_row = &mut result[mutation_index * var_params.num_clusters
                ..(mutation_index + 1) * var_params.num_clusters];
            let data_row = ArrayView1::from(
                &data_preproc.z_update_data[mutation_index * contraction_axis_size
                    ..(mutation_index + 1) * contraction_axis_size],
            );
            result_row.fill(0.0);
            for (contraction_index, data_value) in data_row.iter().enumerate() {
                let theta_row = reshaped_theta.row(contraction_index);
                for cluster_index in 0..var_params.num_clusters {
                    result_row[cluster_index] += *data_value * theta_row[cluster_index];
                }
            }
        }
    }

    result
}

fn sum_log_p_data_theta_with_z(
    theta: &[f64],
    var_params: &VariationalParameters,
    data_preproc: &DataPreprocessor,
    scratch: &mut InferenceScratch,
) -> f64 {
    let contraction_axis_size = var_params.num_dims * var_params.num_grid_points;
    reshape_theta_for_z(theta, var_params, &mut scratch.reshaped_theta);

    let reshaped_theta = ArrayView2::from_shape(
        (contraction_axis_size, var_params.num_clusters),
        &scratch.reshaped_theta,
    )
    .expect("reshaped theta view must match backing storage");
    if data_preproc.use_parallel {
        data_preproc
            .z_update_data
            .par_chunks(contraction_axis_size)
            .enumerate()
            .map(|(mutation_index, data_row)| {
                let row_start = mutation_index * var_params.num_clusters;
                let z_row =
                    ArrayView1::from(&var_params.z[row_start..row_start + var_params.num_clusters]);
                let data_row = ArrayView1::from(data_row);
                let mut row_total = 0.0;
                for (contraction_index, data_value) in data_row.iter().enumerate() {
                    let theta_row = reshaped_theta.row(contraction_index);
                    for cluster_index in 0..var_params.num_clusters {
                        row_total += *data_value * theta_row[cluster_index] * z_row[cluster_index];
                    }
                }
                row_total
            })
            .sum()
    } else {
        let mut total_sum = 0.0;
        for mutation_index in 0..data_preproc.z_update_shape {
            let data_row = ArrayView1::from(
                &data_preproc.z_update_data[mutation_index * contraction_axis_size
                    ..(mutation_index + 1) * contraction_axis_size],
            );
            let row_start = mutation_index * var_params.num_clusters;
            let z_row =
                ArrayView1::from(&var_params.z[row_start..row_start + var_params.num_clusters]);
            for (contraction_index, data_value) in data_row.iter().enumerate() {
                let theta_row = reshaped_theta.row(contraction_index);
                for cluster_index in 0..var_params.num_clusters {
                    total_sum += *data_value * theta_row[cluster_index] * z_row[cluster_index];
                }
            }
        }
        total_sum
    }
}

fn fill_log_p_data_z<'a>(
    var_params: &VariationalParameters,
    data_preproc: &DataPreprocessor,
    scratch: &'a mut InferenceScratch,
) -> &'a mut [f64] {
    let contraction_axis_size = var_params.num_dims * var_params.num_grid_points;

    let transposed_z = &scratch.transposed_z;
    let theta_update_data = ArrayView2::from_shape(
        (var_params.num_data_points, contraction_axis_size),
        &data_preproc.theta_update_data,
    )
    .expect("theta update data shape must match backing storage");
    let result = &mut scratch.log_p_data_z;

    if data_preproc.use_parallel {
        result
            .par_chunks_mut(contraction_axis_size)
            .enumerate()
            .for_each(|(cluster_index, result_chunk)| {
                let z_row = ArrayView1::from(
                    &transposed_z[cluster_index * var_params.num_data_points
                        ..(cluster_index + 1) * var_params.num_data_points],
                );
                result_chunk.fill(0.0);
                for (data_point_index, weight) in z_row.iter().enumerate() {
                    let theta_row = theta_update_data.row(data_point_index);
                    for contraction_index in 0..contraction_axis_size {
                        result_chunk[contraction_index] += *weight * theta_row[contraction_index];
                    }
                }
            });
    } else {
        for cluster_index in 0..var_params.num_clusters {
            let z_row = ArrayView1::from(
                &transposed_z[cluster_index * var_params.num_data_points
                    ..(cluster_index + 1) * var_params.num_data_points],
            );
            let result_chunk = &mut result[cluster_index * contraction_axis_size
                ..(cluster_index + 1) * contraction_axis_size];
            result_chunk.fill(0.0);
            for (data_point_index, weight) in z_row.iter().enumerate() {
                let theta_row = theta_update_data.row(data_point_index);
                for contraction_index in 0..contraction_axis_size {
                    result_chunk[contraction_index] += *weight * theta_row[contraction_index];
                }
            }
        }
    }

    result
}

impl Priors {
    pub fn new(
        num_clusters: usize,
        num_grid_points: usize,
        mix_weight_prior: f64,
    ) -> Result<Self, String> {
        if num_clusters == 0 {
            return Err("num_clusters must be > 0".to_string());
        }
        if num_grid_points == 0 {
            return Err("num_grid_points must be > 0".to_string());
        }
        if mix_weight_prior <= 0.0 {
            return Err("mix_weight_prior must be > 0".to_string());
        }

        let pi = vec![mix_weight_prior; num_clusters];
        let theta_fill = 1.0 / num_grid_points as f64;
        let theta = vec![theta_fill; num_grid_points];
        let log_theta = theta.iter().map(|value| value.ln()).collect::<Vec<_>>();
        let pi_sum = pi.iter().sum::<f64>();
        let pi_log_gamma = ln_gamma(pi_sum) - pi.iter().map(|value| ln_gamma(*value)).sum::<f64>();

        Ok(Self {
            pi,
            theta,
            log_theta,
            pi_log_gamma,
        })
    }
}

impl VariationalParameters {
    fn sample_simplex(size: usize, rng: &mut impl Rng) -> Result<Vec<f64>, String> {
        if size == 0 {
            return Err("simplex size must be > 0".to_string());
        }

        let gamma =
            Gamma::new(1.0, 1.0).map_err(|_| "failed to initialize gamma sampler".to_string())?;
        let mut values = Vec::with_capacity(size);
        for _ in 0..size {
            values.push(gamma.sample(rng));
        }
        let total: f64 = values.iter().sum();
        if total <= 0.0 {
            return Err("sampled simplex has non-positive sum".to_string());
        }
        for value in &mut values {
            *value /= total;
        }
        Ok(values)
    }

    #[allow(dead_code)]
    pub fn new_uniform(
        num_clusters: usize,
        num_data_points: usize,
        num_dims: usize,
        num_grid_points: usize,
    ) -> Result<Self, String> {
        if num_clusters == 0 || num_data_points == 0 || num_dims == 0 || num_grid_points == 0 {
            return Err("all dimensions must be > 0".to_string());
        }

        let pi = vec![1.0 / num_clusters as f64; num_clusters];
        let theta = vec![1.0 / num_grid_points as f64; num_clusters * num_dims * num_grid_points];
        let z = vec![1.0 / num_clusters as f64; num_data_points * num_clusters];

        Self::from_parts(
            pi,
            theta,
            z,
            num_data_points,
            num_clusters,
            num_dims,
            num_grid_points,
        )
    }

    pub fn new_random(
        num_clusters: usize,
        num_data_points: usize,
        num_dims: usize,
        num_grid_points: usize,
        rng: &mut impl Rng,
    ) -> Result<Self, String> {
        if num_clusters == 0 || num_data_points == 0 || num_dims == 0 || num_grid_points == 0 {
            return Err("all dimensions must be > 0".to_string());
        }

        let pi = Self::sample_simplex(num_clusters, rng)?;

        let mut theta = vec![0.0; num_clusters * num_dims * num_grid_points];
        for cluster_index in 0..num_clusters {
            for dim_index in 0..num_dims {
                let simplex = Self::sample_simplex(num_grid_points, rng)?;
                let offset = (cluster_index * num_dims + dim_index) * num_grid_points;
                theta[offset..offset + num_grid_points].copy_from_slice(&simplex);
            }
        }

        let mut z = vec![0.0; num_data_points * num_clusters];
        for data_point_index in 0..num_data_points {
            let simplex = Self::sample_simplex(num_clusters, rng)?;
            let offset = data_point_index * num_clusters;
            z[offset..offset + num_clusters].copy_from_slice(&simplex);
        }

        Self::from_parts(
            pi,
            theta,
            z,
            num_data_points,
            num_clusters,
            num_dims,
            num_grid_points,
        )
    }

    pub fn from_parts(
        pi: Vec<f64>,
        theta: Vec<f64>,
        z: Vec<f64>,
        num_data_points: usize,
        num_clusters: usize,
        num_dims: usize,
        num_grid_points: usize,
    ) -> Result<Self, String> {
        if pi.len() != num_clusters {
            return Err("pi length must equal num_clusters".to_string());
        }
        if z.len() != num_data_points * num_clusters {
            return Err("z length must equal num_data_points * num_clusters".to_string());
        }
        if theta.len() != num_clusters * num_dims * num_grid_points {
            return Err(
                "theta length must equal num_clusters * num_dims * num_grid_points".to_string(),
            );
        }

        Ok(Self {
            pi,
            theta,
            z,
            num_clusters,
            num_data_points,
            num_dims,
            num_grid_points,
        })
    }

    pub fn update_pi(&mut self, priors: &Priors) -> Result<(), String> {
        if priors.pi.len() != self.num_clusters {
            return Err("priors.pi length must equal num_clusters".to_string());
        }

        for cluster_index in 0..self.num_clusters {
            let mut total = priors.pi[cluster_index];
            for data_point_index in 0..self.num_data_points {
                let z_index = data_point_index * self.num_clusters + cluster_index;
                total += self.z[z_index];
            }
            self.pi[cluster_index] = total;
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn update_z(&mut self, data_preproc: &DataPreprocessor) -> Result<(), String> {
        let mut scratch = InferenceScratch::new(self, data_preproc);
        self.update_z_with_profile(data_preproc, &mut scratch, None)
    }

    fn update_z_with_profile(
        &mut self,
        data_preproc: &DataPreprocessor,
        scratch: &mut InferenceScratch,
        detail: Option<&mut FitDetailProfile>,
    ) -> Result<(), String> {
        if data_preproc.z_update_shape != self.num_data_points {
            return Err(
                "data_preprocessor mutation dimension must equal num_data_points".to_string(),
            );
        }

        let started = Instant::now();
        fill_log_p_data_theta(&self.theta, self, data_preproc, scratch);
        let update_z_contract = started.elapsed();
        let new_z = &mut scratch.log_p_data_theta;
        let pi_sum = self.pi.iter().sum::<f64>();
        let psi_sum = digamma(pi_sum);
        let psi_term = self
            .pi
            .iter()
            .map(|value| digamma(*value) - psi_sum)
            .collect::<Vec<_>>();

        let started = Instant::now();
        if data_preproc.use_parallel {
            self.z
                .par_chunks_mut(self.num_clusters)
                .zip(new_z.par_chunks_mut(self.num_clusters))
                .for_each(|(z_row, new_z_row)| {
                    for cluster_index in 0..z_row.len() {
                        new_z_row[cluster_index] += psi_term[cluster_index];
                    }
                    let row_norm = log_sum_exp(new_z_row);
                    for cluster_index in 0..z_row.len() {
                        z_row[cluster_index] = (new_z_row[cluster_index] - row_norm).exp();
                    }
                });
        } else {
            let psi_term = ArrayView1::from(&psi_term);
            let mut z_view =
                ArrayViewMut2::from_shape((self.num_data_points, self.num_clusters), &mut self.z)
                    .expect("z shape must match backing storage");
            let mut new_z_view =
                ArrayViewMut2::from_shape((self.num_data_points, self.num_clusters), new_z)
                    .expect("new z shape must match backing storage");

            for (mut z_row, mut new_z_row) in
                z_view.outer_iter_mut().zip(new_z_view.outer_iter_mut())
            {
                for (value, psi) in new_z_row.iter_mut().zip(psi_term.iter()) {
                    *value += *psi;
                }
                let row_norm = log_sum_exp(new_z_row.as_slice().expect("row view is contiguous"));
                for (z_value, log_value) in z_row.iter_mut().zip(new_z_row.iter()) {
                    *z_value = (*log_value - row_norm).exp();
                }
            }
        }

        let update_z_normalize = started.elapsed();
        refresh_z_views(self, scratch);
        if let Some(detail) = detail {
            detail.update_z_contract += update_z_contract;
            detail.update_z_normalize += update_z_normalize;
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub fn update_theta(
        &mut self,
        priors: &Priors,
        data_preproc: &DataPreprocessor,
    ) -> Result<(), String> {
        let mut scratch = InferenceScratch::new(self, data_preproc);
        self.update_theta_with_profile(priors, data_preproc, &mut scratch, None)
    }

    fn update_theta_with_profile(
        &mut self,
        priors: &Priors,
        data_preproc: &DataPreprocessor,
        scratch: &mut InferenceScratch,
        detail: Option<&mut FitDetailProfile>,
    ) -> Result<(), String> {
        if priors.log_theta.len() != self.num_grid_points {
            return Err("priors.log_theta length must equal num_grid_points".to_string());
        }
        if data_preproc.theta_update_shape != (self.num_dims, self.num_grid_points) {
            return Err(
                "data_preprocessor theta shape must match variational dimensions".to_string(),
            );
        }

        let started = Instant::now();
        fill_log_p_data_z(self, data_preproc, scratch);
        let update_theta_contract = started.elapsed();
        let log_p_data_z = &mut scratch.log_p_data_z;

        let started = Instant::now();
        let e_log_p_data_theta = if data_preproc.use_parallel {
            self.theta
                .par_chunks_mut(self.num_grid_points)
                .zip(log_p_data_z.par_chunks_mut(self.num_grid_points))
                .map(|(theta_row, log_row)| {
                    let mut row_data_term = 0.0;
                    for (grid_index, lr) in log_row.iter_mut().enumerate().take(theta_row.len()) {
                        *lr += priors.log_theta[grid_index];
                    }
                    let row_norm = log_sum_exp(log_row);
                    for grid_index in 0..theta_row.len() {
                        let theta_value = (log_row[grid_index] - row_norm).exp();
                        theta_row[grid_index] = theta_value;
                        row_data_term +=
                            theta_value * (log_row[grid_index] - priors.log_theta[grid_index]);
                    }
                    row_data_term
                })
                .sum()
        } else {
            let prior_log_theta = ArrayView1::from(&priors.log_theta);
            let mut theta_view = ArrayViewMut2::from_shape(
                (self.num_clusters * self.num_dims, self.num_grid_points),
                &mut self.theta,
            )
            .expect("theta shape must match backing storage");
            let mut log_p_data_z_view = ArrayViewMut2::from_shape(
                (self.num_clusters * self.num_dims, self.num_grid_points),
                log_p_data_z,
            )
            .expect("log_p_data_z shape must match backing storage");
            let mut total_data_term = 0.0;

            for (mut theta_row, mut log_row) in theta_view
                .outer_iter_mut()
                .zip(log_p_data_z_view.outer_iter_mut())
            {
                for (value, prior) in log_row.iter_mut().zip(prior_log_theta.iter()) {
                    *value += *prior;
                }

                let row_norm = log_sum_exp(log_row.as_slice().expect("row view is contiguous"));
                for ((theta_value, log_value), prior) in theta_row
                    .iter_mut()
                    .zip(log_row.iter())
                    .zip(prior_log_theta.iter())
                {
                    let normalized = (*log_value - row_norm).exp();
                    *theta_value = normalized;
                    total_data_term += normalized * (*log_value - *prior);
                }
            }
            total_data_term
        };

        let update_theta_normalize = started.elapsed();
        scratch.cached_e_log_p_data_theta = Some(e_log_p_data_theta);
        if let Some(detail) = detail {
            detail.update_theta_contract += update_theta_contract;
            detail.update_theta_normalize += update_theta_normalize;
        }

        Ok(())
    }
}

#[allow(dead_code)]
pub fn compute_e_log_p(
    priors: &Priors,
    var_params: &VariationalParameters,
    data_preproc: &DataPreprocessor,
) -> Result<f64, String> {
    let mut scratch = InferenceScratch::new(var_params, data_preproc);
    compute_e_log_p_with_profile(priors, var_params, data_preproc, &mut scratch, None)
}

fn compute_e_log_p_with_profile(
    priors: &Priors,
    var_params: &VariationalParameters,
    data_preproc: &DataPreprocessor,
    scratch: &mut InferenceScratch,
    detail: Option<&mut FitDetailProfile>,
) -> Result<f64, String> {
    let mut log_p = priors.pi_log_gamma;

    let pi_sum = var_params.pi.iter().sum::<f64>();
    let psi_sum = digamma(pi_sum);
    let started = Instant::now();
    let e_log_p_z_sums = started.elapsed();

    let started = Instant::now();
    for cluster_index in 0..var_params.num_clusters {
        let p_pi_z_term = priors.pi[cluster_index] + scratch.z_sums[cluster_index] - 1.0;
        let pi_psi_term = digamma(var_params.pi[cluster_index]) - psi_sum;
        log_p += p_pi_z_term * pi_psi_term;
    }
    let e_log_p_pi_term = started.elapsed();

    let started = Instant::now();
    let theta_view = ArrayView2::from_shape(
        (
            var_params.num_clusters * var_params.num_dims,
            var_params.num_grid_points,
        ),
        &var_params.theta,
    )
    .expect("theta shape must match backing storage");
    let prior_log_theta = ArrayView1::from(&priors.log_theta);
    log_p += theta_view
        .outer_iter()
        .map(|theta_row| theta_row.dot(&prior_log_theta))
        .sum::<f64>();
    let e_log_p_theta_prior = started.elapsed();

    let started = Instant::now();
    let e_log_p_data_theta_value = if let Some(value) = scratch.cached_e_log_p_data_theta {
        value
    } else {
        let value =
            sum_log_p_data_theta_with_z(&var_params.theta, var_params, data_preproc, scratch);
        scratch.cached_e_log_p_data_theta = Some(value);
        value
    };
    log_p += e_log_p_data_theta_value;
    let e_log_p_data_theta = started.elapsed();
    let e_log_p_data_accum = Duration::ZERO;

    if let Some(detail) = detail {
        detail.e_log_p_z_sums += e_log_p_z_sums;
        detail.e_log_p_pi_term += e_log_p_pi_term;
        detail.e_log_p_theta_prior += e_log_p_theta_prior;
        detail.e_log_p_data_theta += e_log_p_data_theta;
        detail.e_log_p_data_accum += e_log_p_data_accum;
    }

    Ok(log_p)
}

pub fn compute_e_log_q(var_params: &VariationalParameters, eps: f64) -> Result<f64, String> {
    compute_e_log_q_with_profile(var_params, eps, None)
}

pub fn compute_e_log_q_with_profile(
    var_params: &VariationalParameters,
    eps: f64,
    detail: Option<&mut FitDetailProfile>,
) -> Result<f64, String> {
    if eps <= 0.0 {
        return Err("eps must be > 0".to_string());
    }

    let mut log_q = 0.0;

    let started = Instant::now();
    let pi_sum = var_params.pi.iter().sum::<f64>();
    log_q += ln_gamma(pi_sum) - var_params.pi.iter().map(|v| ln_gamma(*v)).sum::<f64>();

    let psi_sum = digamma(pi_sum);
    for cluster_index in 0..var_params.num_clusters {
        let psi_term = digamma(var_params.pi[cluster_index]) - psi_sum;
        log_q += psi_term * (var_params.pi[cluster_index] - 1.0);
    }
    let e_log_q_pi_term = started.elapsed();

    let started = Instant::now();
    for value in &var_params.theta {
        log_q += value * (value + eps).ln();
    }
    let e_log_q_theta_term = started.elapsed();

    let started = Instant::now();
    log_q += ArrayView1::from(&var_params.z)
        .iter()
        .map(|value| value * (value + eps).ln())
        .sum::<f64>();
    let e_log_q_z_term = started.elapsed();

    if let Some(detail) = detail {
        detail.e_log_q_pi_term += e_log_q_pi_term;
        detail.e_log_q_theta_term += e_log_q_theta_term;
        detail.e_log_q_z_term += e_log_q_z_term;
    }

    Ok(log_q)
}

#[allow(dead_code)]
pub fn compute_elbo(
    priors: &Priors,
    var_params: &VariationalParameters,
    data_preproc: &DataPreprocessor,
    eps: f64,
) -> Result<f64, String> {
    let mut scratch = InferenceScratch::new(var_params, data_preproc);
    compute_elbo_with_profile(priors, var_params, data_preproc, eps, &mut scratch, None)
}

fn compute_elbo_with_profile(
    priors: &Priors,
    var_params: &VariationalParameters,
    data_preproc: &DataPreprocessor,
    eps: f64,
    scratch: &mut InferenceScratch,
    detail: Option<&mut FitDetailProfile>,
) -> Result<f64, String> {
    if let Some(detail) = detail {
        let started = Instant::now();
        let e_log_p =
            compute_e_log_p_with_profile(priors, var_params, data_preproc, scratch, Some(detail))?;
        detail.elbo_e_log_p += started.elapsed();

        let started = Instant::now();
        let e_log_q = compute_e_log_q_with_profile(var_params, eps, Some(detail))?;
        detail.elbo_e_log_q += started.elapsed();

        Ok(e_log_p - e_log_q)
    } else {
        Ok(
            compute_e_log_p_with_profile(priors, var_params, data_preproc, scratch, None)?
                - compute_e_log_q(var_params, eps)?,
        )
    }
}

pub fn fit_variational_model(
    priors: &Priors,
    var_params: &mut VariationalParameters,
    data_preproc: &DataPreprocessor,
    convergence_threshold: f64,
    max_iters: usize,
) -> Result<Vec<f64>, String> {
    fit_variational_model_with_profile(
        priors,
        var_params,
        data_preproc,
        convergence_threshold,
        max_iters,
    )
    .map(|(trace, _)| trace)
}

pub fn fit_variational_model_with_profile(
    priors: &Priors,
    var_params: &mut VariationalParameters,
    data_preproc: &DataPreprocessor,
    convergence_threshold: f64,
    max_iters: usize,
) -> Result<(Vec<f64>, FitProfile), String> {
    if convergence_threshold <= 0.0 {
        return Err("convergence_threshold must be > 0".to_string());
    }
    if max_iters == 0 {
        return Err("max_iters must be > 0".to_string());
    }

    let eps = 1e-6;
    let mut profile = FitProfile::default();
    let mut detail = FitDetailProfile::default();
    let mut scratch = InferenceScratch::new(var_params, data_preproc);
    let started = Instant::now();
    let mut elbo_trace = vec![compute_elbo_with_profile(
        priors,
        var_params,
        data_preproc,
        eps,
        &mut scratch,
        Some(&mut detail),
    )?];
    profile.initial_elbo = started.elapsed();

    for _ in 0..max_iters {
        let started = Instant::now();
        var_params.update_z_with_profile(data_preproc, &mut scratch, Some(&mut detail))?;
        profile.update_z += started.elapsed();

        let started = Instant::now();
        var_params.update_pi(priors)?;
        profile.update_pi += started.elapsed();

        let started = Instant::now();
        var_params.update_theta_with_profile(
            priors,
            data_preproc,
            &mut scratch,
            Some(&mut detail),
        )?;
        profile.update_theta += started.elapsed();

        let started = Instant::now();
        let curr_elbo = compute_elbo_with_profile(
            priors,
            var_params,
            data_preproc,
            eps,
            &mut scratch,
            Some(&mut detail),
        )?;
        profile.iter_elbo += started.elapsed();
        let prev_elbo = *elbo_trace.last().expect("non-empty");
        elbo_trace.push(curr_elbo);
        profile.iterations += 1;

        let denom = curr_elbo.abs().max(1e-12);
        let diff = (curr_elbo - prev_elbo) / denom;
        if diff < convergence_threshold {
            break;
        }
    }

    if std::env::var_os("PCV_PROFILE").is_some() {
        profile.print_summary();
        detail.print_summary();
    }

    Ok((elbo_trace, profile))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LogLikelihoodTensor;
    use rand::SeedableRng;

    fn approx_eq(left: f64, right: f64, tol: f64) {
        let delta = (left - right).abs();
        assert!(
            delta < tol,
            "left={left}, right={right}, delta={delta}, tol={tol}"
        );
    }

    #[test]
    fn priors_match_python_shapes() {
        let priors = Priors::new(3, 4, 1.0).unwrap();

        assert_eq!(priors.pi, vec![1.0, 1.0, 1.0]);
        assert_eq!(priors.theta, vec![0.25, 0.25, 0.25, 0.25]);
        assert_eq!(priors.log_theta.len(), 4);
        approx_eq(priors.pi_log_gamma, ln_gamma(3.0), 1e-12);
    }

    #[test]
    fn update_pi_matches_row_sum_formula() {
        let theta = vec![1.0 / 5.0; 3 * 2 * 5];
        let z = vec![0.2, 0.3, 0.5, 0.1, 0.8, 0.1, 0.7, 0.2, 0.1];

        let error =
            VariationalParameters::from_parts(vec![0.2, 0.2], theta, z, 3, 3, 2, 5).unwrap_err();

        assert_eq!(error, "pi length must equal num_clusters");
    }

    #[test]
    fn update_pi_computes_prior_plus_column_sums() {
        let priors = Priors::new(3, 5, 1.0).unwrap();
        let theta = vec![1.0 / 5.0; 3 * 2 * 5];
        let z = vec![0.2, 0.3, 0.5, 0.1, 0.8, 0.1, 0.7, 0.2, 0.1];

        let mut var_params =
            VariationalParameters::from_parts(vec![0.3, 0.3, 0.4], theta, z, 3, 3, 2, 5).unwrap();

        var_params.update_pi(&priors).unwrap();

        approx_eq(var_params.pi[0], 2.0, 1e-12);
        approx_eq(var_params.pi[1], 2.3, 1e-12);
        approx_eq(var_params.pi[2], 1.7, 1e-12);
    }

    #[test]
    fn uniform_initializer_sets_expected_shapes() {
        let var_params = VariationalParameters::new_uniform(3, 4, 2, 5).unwrap();

        assert_eq!(var_params.pi.len(), 3);
        assert_eq!(var_params.theta.len(), 30);
        assert_eq!(var_params.z.len(), 12);
        approx_eq(var_params.pi[0], 1.0 / 3.0, 1e-12);
        approx_eq(var_params.theta[0], 1.0 / 5.0, 1e-12);
        approx_eq(var_params.z[0], 1.0 / 3.0, 1e-12);
    }

    #[test]
    fn random_initializer_is_seed_reproducible() {
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(42);

        let a = VariationalParameters::new_random(3, 4, 2, 5, &mut rng_a).unwrap();
        let b = VariationalParameters::new_random(3, 4, 2, 5, &mut rng_b).unwrap();

        assert_eq!(a.pi, b.pi);
        assert_eq!(a.theta, b.theta);
        assert_eq!(a.z, b.z);
    }

    #[test]
    fn data_preprocessor_reshapes_log_p_data_for_z_update() {
        let tensor = LogLikelihoodTensor {
            num_mutations: 2,
            num_samples: 2,
            num_grid_points: 3,
            values: vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        };

        let preproc = DataPreprocessor::new(&tensor, false);

        assert_eq!(preproc.theta_update_data, tensor.values);
        assert_eq!(
            preproc.z_update_data,
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 10.0, 8.0, 11.0, 9.0, 12.0]
        );
        assert_eq!(preproc.theta_update_shape, (2, 3));
        assert_eq!(preproc.z_update_shape, 2);
    }

    #[test]
    fn update_z_normalizes_rows() {
        let tensor = LogLikelihoodTensor {
            num_mutations: 2,
            num_samples: 2,
            num_grid_points: 2,
            values: vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
        };
        let preproc = DataPreprocessor::new(&tensor, false);
        let mut var_params = VariationalParameters::from_parts(
            vec![1.5, 2.5],
            vec![0.7, 0.3, 0.4, 0.6, 0.2, 0.8, 0.5, 0.5],
            vec![0.5, 0.5, 0.5, 0.5],
            2,
            2,
            2,
            2,
        )
        .unwrap();

        var_params.update_z(&preproc).unwrap();

        let row0 = var_params.z[0] + var_params.z[1];
        let row1 = var_params.z[2] + var_params.z[3];
        approx_eq(row0, 1.0, 1e-12);
        approx_eq(row1, 1.0, 1e-12);
        assert!(var_params.z.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn update_theta_normalizes_each_cluster_dim_row() {
        let tensor = LogLikelihoodTensor {
            num_mutations: 2,
            num_samples: 2,
            num_grid_points: 2,
            values: vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
        };
        let preproc = DataPreprocessor::new(&tensor, false);
        let priors = Priors::new(2, 2, 1.0).unwrap();
        let mut var_params = VariationalParameters::from_parts(
            vec![1.5, 2.5],
            vec![0.7, 0.3, 0.4, 0.6, 0.2, 0.8, 0.5, 0.5],
            vec![0.6, 0.4, 0.3, 0.7],
            2,
            2,
            2,
            2,
        )
        .unwrap();

        var_params.update_theta(&priors, &preproc).unwrap();

        for cluster_index in 0..2 {
            for dim_index in 0..2 {
                let row_start = (cluster_index * 2 + dim_index) * 2;
                let row_sum = var_params.theta[row_start] + var_params.theta[row_start + 1];
                approx_eq(row_sum, 1.0, 1e-12);
            }
        }
        assert!(var_params.theta.iter().all(|value| *value > 0.0));
    }

    #[test]
    fn computes_finite_elbo() {
        let tensor = LogLikelihoodTensor {
            num_mutations: 2,
            num_samples: 2,
            num_grid_points: 2,
            values: vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5],
        };
        let preproc = DataPreprocessor::new(&tensor, false);
        let priors = Priors::new(2, 2, 1.0).unwrap();
        let var_params = VariationalParameters::new_uniform(2, 2, 2, 2).unwrap();

        let elbo = compute_elbo(&priors, &var_params, &preproc, 1e-6).unwrap();
        assert!(elbo.is_finite());
    }

    #[test]
    fn runs_variational_loop_and_collects_trace() {
        let tensor = LogLikelihoodTensor {
            num_mutations: 3,
            num_samples: 2,
            num_grid_points: 2,
            values: vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.3, 0.7, 0.7, 0.3],
        };
        let preproc = DataPreprocessor::new(&tensor, false);
        let priors = Priors::new(2, 2, 1.0).unwrap();
        let mut var_params = VariationalParameters::new_uniform(2, 3, 2, 2).unwrap();

        let trace = fit_variational_model(&priors, &mut var_params, &preproc, 1e-6, 10).unwrap();

        assert!(!trace.is_empty());
        assert!(trace.iter().all(|value| value.is_finite()));
    }
}
