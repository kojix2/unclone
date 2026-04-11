use std::collections::{BTreeSet, HashMap};
use std::env;
use std::time::{Duration, Instant};

use crate::abi::{PcvConfig, PcvResult, PcvRow};
use crate::inference::fit_variational_model;
use crate::preprocess::{build_log_p_data, build_log_p_data_parallel, get_ccf_grid};
use crate::types::{DataPreprocessor, Density, Priors, VariationalParameters};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

#[derive(Clone, Copy)]
pub(crate) struct KernelShape {
    pub(crate) num_mutations: usize,
    pub(crate) num_samples: usize,
}

pub(crate) struct BorrowedCall<'a, TConfig> {
    pub(crate) cfg: &'a TConfig,
    pub(crate) input_rows: &'a [PcvRow],
    pub(crate) shape: KernelShape,
}

impl<'a, TConfig> BorrowedCall<'a, TConfig> {
    pub(crate) unsafe fn borrow(
        config: *const TConfig,
        rows: *const PcvRow,
        rows_len: usize,
        num_mutations: usize,
        num_samples: usize,
    ) -> Result<Self, String> {
        if config.is_null() || rows.is_null() {
            return Err("config or rows is null".to_string());
        }
        if rows_len == 0 || num_mutations == 0 || num_samples == 0 {
            return Err("rows_len, num_mutations, num_samples must be > 0".to_string());
        }

        Ok(Self {
            cfg: &*config,
            input_rows: std::slice::from_raw_parts(rows, rows_len),
            shape: KernelShape {
                num_mutations,
                num_samples,
            },
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct VariationalCompatInit {
    pub(crate) compat_pi: *const f64,
    pub(crate) compat_pi_len: usize,
    pub(crate) compat_theta: *const f64,
    pub(crate) compat_theta_len: usize,
    pub(crate) compat_z: *const f64,
    pub(crate) compat_z_len: usize,
}

#[derive(Default, Debug, Clone, PartialEq)]
struct KernelProfile {
    ccf_grid: Duration,
    log_p_data: Duration,
    priors: Duration,
    data_preproc: Duration,
    restarts: Duration,
    result_build: Duration,
    total: Duration,
}

impl KernelProfile {
    fn print_summary(
        &self,
        kernel_threads: usize,
        restart_parallelism: usize,
        num_restarts: usize,
        num_mutations: usize,
        num_samples: usize,
    ) {
        eprintln!(
            "[tyclone-kernel-profile] kernel_threads={} restart_parallelism={} num_restarts={} num_mutations={} num_samples={} ccf_grid_ms={:.3} log_p_data_ms={:.3} priors_ms={:.3} data_preproc_ms={:.3} restarts_ms={:.3} result_build_ms={:.3} total_ms={:.3}",
            kernel_threads,
            restart_parallelism,
            num_restarts,
            num_mutations,
            num_samples,
            self.ccf_grid.as_secs_f64() * 1_000.0,
            self.log_p_data.as_secs_f64() * 1_000.0,
            self.priors.as_secs_f64() * 1_000.0,
            self.data_preproc.as_secs_f64() * 1_000.0,
            self.restarts.as_secs_f64() * 1_000.0,
            self.result_build.as_secs_f64() * 1_000.0,
            self.total.as_secs_f64() * 1_000.0,
        );
    }
}

pub(crate) fn run_vi_request(
    request: BorrowedCall<'_, PcvConfig>,
    compat: VariationalCompatInit,
) -> Result<PcvResult, String> {
    let cfg = request.cfg;
    let input_rows = request.input_rows;
    let num_mutations = request.shape.num_mutations;
    let num_samples = request.shape.num_samples;
    let profiling_enabled = env::var_os("PCV_PROFILE").is_some();
    let total_started = Instant::now();
    let mut kernel_profile = KernelProfile::default();
    let density = Density::try_from(cfg.density)?;

    if cfg.num_grid_points < 2 {
        return Err("num_grid_points must be >= 2".to_string());
    }
    if cfg.num_clusters <= 0 {
        return Err("num_clusters must be > 0".to_string());
    }
    if cfg.num_restarts <= 0 {
        return Err("num_restarts must be > 0".to_string());
    }
    if cfg.kernel_threads < 0 {
        return Err("kernel_threads must be >= 0".to_string());
    }

    let kernel_threads = if cfg.kernel_threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        cfg.kernel_threads as usize
    };

    let restart_parallelism = if cfg.restart_parallelism <= 0 {
        1
    } else {
        cfg.restart_parallelism as usize
    };
    let enable_kernel_parallel = kernel_threads > 1;
    let enable_restart_parallel = restart_parallelism > 1 && cfg.num_restarts > 1;
    let rayon_pool_threads = if enable_restart_parallel {
        kernel_threads.max(restart_parallelism)
    } else {
        kernel_threads
    };

    let rayon_pool = if enable_kernel_parallel || enable_restart_parallel {
        Some(
            ThreadPoolBuilder::new()
                .num_threads(rayon_pool_threads)
                .build()
                .map_err(|error| format!("failed to build rayon thread pool: {error}"))?,
        )
    } else {
        None
    };

    let started = Instant::now();
    let ccf_grid = get_ccf_grid(cfg.num_grid_points as usize, 1e-6)?;
    kernel_profile.ccf_grid = started.elapsed();

    let started = Instant::now();
    let log_p_data = if enable_kernel_parallel {
        let pool = rayon_pool
            .as_ref()
            .expect("rayon pool should exist when kernel parallelism is enabled");
        pool.install(|| {
            build_log_p_data_parallel(
                input_rows,
                num_mutations,
                num_samples,
                &ccf_grid,
                density,
                cfg.precision,
            )
        })?
    } else {
        build_log_p_data(
            input_rows,
            num_mutations,
            num_samples,
            &ccf_grid,
            density,
            cfg.precision,
        )?
    };
    kernel_profile.log_p_data = started.elapsed();

    let started = Instant::now();
    let priors = Priors::new(
        cfg.num_clusters as usize,
        cfg.num_grid_points as usize,
        cfg.mix_weight_prior,
    )?;
    kernel_profile.priors = started.elapsed();

    let started = Instant::now();
    let data_preproc = DataPreprocessor::new(&log_p_data, enable_kernel_parallel);
    kernel_profile.data_preproc = started.elapsed();

    let base_seed = if cfg.use_seed == 1 {
        cfg.seed
    } else {
        rand::random::<u64>()
    };

    let mut compat_var_params_list: Option<Vec<VariationalParameters>> = decode_compat_var_params(
        cfg,
        log_p_data.num_mutations,
        log_p_data.num_samples,
        compat.compat_pi,
        compat.compat_pi_len,
        compat.compat_theta,
        compat.compat_theta_len,
        compat.compat_z,
        compat.compat_z_len,
    )?;

    let mut run_all_restarts = || -> Result<Vec<RestartOutcome>, String> {
        if let Some(var_params_list) = compat_var_params_list.take() {
            return var_params_list
                .into_iter()
                .enumerate()
                .map(|(i, vp)| {
                    run_restart_with_var_params(
                        i,
                        base_seed,
                        vp,
                        &priors,
                        &data_preproc,
                        cfg.convergence_threshold,
                        cfg.max_iters as usize,
                    )
                })
                .collect();
        }

        let restart_range = 0..(cfg.num_restarts as usize);

        if !enable_restart_parallel {
            let mut shared_restart_rng = StdRng::seed_from_u64(base_seed);
            restart_range
                .map(|restart| {
                    run_restart_with_rng(
                        restart,
                        base_seed,
                        cfg.num_clusters as usize,
                        log_p_data.num_mutations,
                        log_p_data.num_samples,
                        log_p_data.num_grid_points,
                        &priors,
                        &data_preproc,
                        cfg.convergence_threshold,
                        cfg.max_iters as usize,
                        &mut shared_restart_rng,
                    )
                })
                .collect()
        } else {
            let pool = rayon_pool
                .as_ref()
                .ok_or_else(|| "rayon pool missing for parallel restart execution".to_string())?;

            pool.install(|| {
                restart_range
                    .into_par_iter()
                    .map(|restart| {
                        run_restart_seeded(
                            restart,
                            base_seed,
                            cfg.num_clusters as usize,
                            log_p_data.num_mutations,
                            log_p_data.num_samples,
                            log_p_data.num_grid_points,
                            &priors,
                            &data_preproc,
                            cfg.convergence_threshold,
                            cfg.max_iters as usize,
                        )
                    })
                    .collect()
            })
        }
    };

    let started = Instant::now();
    let mut restart_outcomes = run_all_restarts()?;
    kernel_profile.restarts = started.elapsed();

    restart_outcomes.sort_by_key(|outcome| outcome.metric.restart_index);

    for outcome in &restart_outcomes {
        if cfg.print_freq > 0 {
            eprintln!(
                "[tyclone] restart={} seed={} final_elbo={} used_clusters={}",
                outcome.metric.restart_index,
                outcome.metric.restart_seed,
                outcome.metric.final_elbo,
                outcome.metric.used_clusters
            );
        }
    }

    let best_outcome_index = best_restart_index(&restart_outcomes)
        .ok_or_else(|| "no restart produced variational parameters".to_string())?;

    let best_elbo = restart_outcomes[best_outcome_index].metric.final_elbo;
    let best_restart_index = restart_outcomes[best_outcome_index].metric.restart_index;

    if cfg.print_freq > 0 {
        eprintln!(
            "[tyclone] best_restart={} best_final_elbo={}",
            best_restart_index, best_elbo
        );
    }

    let best_var_params = restart_outcomes.swap_remove(best_outcome_index).var_params;

    let started = Instant::now();
    let result =
        build_result_from_variational(&best_var_params, &ccf_grid, num_mutations, num_samples)?;
    kernel_profile.result_build = started.elapsed();
    kernel_profile.total = total_started.elapsed();

    if profiling_enabled {
        kernel_profile.print_summary(
            kernel_threads,
            restart_parallelism,
            cfg.num_restarts as usize,
            num_mutations,
            num_samples,
        );
    }

    Ok(result)
}

pub(crate) fn get_output_grid(grid_size: usize) -> Result<Vec<f64>, String> {
    if grid_size < 2 {
        return Err("grid_size must be >= 2".to_string());
    }

    let step = 1.0 / ((grid_size - 1) as f64);
    let mut grid = Vec::with_capacity(grid_size);
    for idx in 0..grid_size {
        grid.push(step * idx as f64);
    }

    Ok(grid)
}

fn num_used_clusters(var_params: &VariationalParameters) -> usize {
    let k = var_params.num_clusters;
    let n = var_params.num_data_points;
    let mut used = vec![false; k];

    for data_point_index in 0..n {
        let row_start = data_point_index * k;
        let row = &var_params.z[row_start..row_start + k];

        let mut best_k = 0usize;
        let mut best_p = f64::NEG_INFINITY;
        for (cluster_index, value) in row.iter().enumerate() {
            if *value > best_p {
                best_p = *value;
                best_k = cluster_index;
            }
        }
        used[best_k] = true;
    }

    used.into_iter().filter(|flag| *flag).count()
}

#[derive(Clone, Debug)]
pub(crate) struct RestartMetric {
    pub(crate) restart_index: usize,
    pub(crate) restart_seed: u64,
    pub(crate) final_elbo: f64,
    pub(crate) used_clusters: usize,
}

pub(crate) struct RestartOutcome {
    pub(crate) metric: RestartMetric,
    pub(crate) var_params: VariationalParameters,
}

fn restart_seed(base_seed: u64, restart_index: usize) -> u64 {
    base_seed ^ ((restart_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

#[allow(clippy::too_many_arguments)]
fn run_restart_with_rng(
    restart_index: usize,
    restart_seed: u64,
    num_clusters: usize,
    log_p_num_mutations: usize,
    log_p_num_samples: usize,
    log_p_num_grid_points: usize,
    priors: &Priors,
    data_preproc: &DataPreprocessor,
    convergence_threshold: f64,
    max_iters: usize,
    restart_rng: &mut impl Rng,
) -> Result<RestartOutcome, String> {
    let mut var_params = VariationalParameters::new_random(
        num_clusters,
        log_p_num_mutations,
        log_p_num_samples,
        log_p_num_grid_points,
        restart_rng,
    )?;

    let trace = fit_variational_model(
        priors,
        &mut var_params,
        data_preproc,
        convergence_threshold,
        max_iters,
    )?;

    let final_elbo = *trace.last().unwrap_or(&f64::NEG_INFINITY);
    let used_clusters = num_used_clusters(&var_params);

    Ok(RestartOutcome {
        metric: RestartMetric {
            restart_index,
            restart_seed,
            final_elbo,
            used_clusters,
        },
        var_params,
    })
}

fn run_restart_with_var_params(
    restart_index: usize,
    restart_seed: u64,
    mut var_params: VariationalParameters,
    priors: &Priors,
    data_preproc: &DataPreprocessor,
    convergence_threshold: f64,
    max_iters: usize,
) -> Result<RestartOutcome, String> {
    let trace = fit_variational_model(
        priors,
        &mut var_params,
        data_preproc,
        convergence_threshold,
        max_iters,
    )?;

    let final_elbo = *trace.last().unwrap_or(&f64::NEG_INFINITY);
    let used_clusters = num_used_clusters(&var_params);

    Ok(RestartOutcome {
        metric: RestartMetric {
            restart_index,
            restart_seed,
            final_elbo,
            used_clusters,
        },
        var_params,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_restart_seeded(
    restart_index: usize,
    base_seed: u64,
    num_clusters: usize,
    log_p_num_mutations: usize,
    log_p_num_samples: usize,
    log_p_num_grid_points: usize,
    priors: &Priors,
    data_preproc: &DataPreprocessor,
    convergence_threshold: f64,
    max_iters: usize,
) -> Result<RestartOutcome, String> {
    let restart_seed = restart_seed(base_seed, restart_index);
    let mut restart_rng = StdRng::seed_from_u64(restart_seed);
    run_restart_with_rng(
        restart_index,
        restart_seed,
        num_clusters,
        log_p_num_mutations,
        log_p_num_samples,
        log_p_num_grid_points,
        priors,
        data_preproc,
        convergence_threshold,
        max_iters,
        &mut restart_rng,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_compat_var_params(
    cfg: &PcvConfig,
    num_mutations: usize,
    num_samples: usize,
    compat_pi: *const f64,
    compat_pi_len: usize,
    compat_theta: *const f64,
    compat_theta_len: usize,
    compat_z: *const f64,
    compat_z_len: usize,
) -> Result<Option<Vec<VariationalParameters>>, String> {
    let any_compat_input = !compat_pi.is_null()
        || !compat_theta.is_null()
        || !compat_z.is_null()
        || compat_pi_len > 0
        || compat_theta_len > 0
        || compat_z_len > 0;

    if !any_compat_input {
        return Ok(None);
    }

    if compat_pi.is_null() || compat_theta.is_null() || compat_z.is_null() {
        return Err(
            "compat init pointers must be non-null when compat init is provided".to_string(),
        );
    }

    let num_restarts = cfg.num_restarts as usize;
    let num_clusters = cfg.num_clusters as usize;
    let num_grid_points = cfg.num_grid_points as usize;

    let expected_pi_len = num_restarts
        .checked_mul(num_clusters)
        .ok_or_else(|| "compat init pi length overflow".to_string())?;
    let expected_theta_len = num_restarts
        .checked_mul(num_clusters)
        .and_then(|v| v.checked_mul(num_samples))
        .and_then(|v| v.checked_mul(num_grid_points))
        .ok_or_else(|| "compat init theta length overflow".to_string())?;
    let expected_z_len = num_restarts
        .checked_mul(num_mutations)
        .and_then(|v| v.checked_mul(num_clusters))
        .ok_or_else(|| "compat init z length overflow".to_string())?;

    if compat_pi_len != expected_pi_len {
        return Err(format!(
            "compat init pi length mismatch: got {}, expected {}",
            compat_pi_len, expected_pi_len
        ));
    }
    if compat_theta_len != expected_theta_len {
        return Err(format!(
            "compat init theta length mismatch: got {}, expected {}",
            compat_theta_len, expected_theta_len
        ));
    }
    if compat_z_len != expected_z_len {
        return Err(format!(
            "compat init z length mismatch: got {}, expected {}",
            compat_z_len, expected_z_len
        ));
    }

    let pi_slice = unsafe { std::slice::from_raw_parts(compat_pi, compat_pi_len) };
    let theta_slice = unsafe { std::slice::from_raw_parts(compat_theta, compat_theta_len) };
    let z_slice = unsafe { std::slice::from_raw_parts(compat_z, compat_z_len) };

    let per_restart_pi = num_clusters;
    let per_restart_theta = num_clusters * num_samples * num_grid_points;
    let per_restart_z = num_mutations * num_clusters;

    let mut list = Vec::with_capacity(num_restarts);
    for restart_index in 0..num_restarts {
        let pi_start = restart_index * per_restart_pi;
        let theta_start = restart_index * per_restart_theta;
        let z_start = restart_index * per_restart_z;

        let pi = pi_slice[pi_start..pi_start + per_restart_pi].to_vec();
        let theta = theta_slice[theta_start..theta_start + per_restart_theta].to_vec();
        let z = z_slice[z_start..z_start + per_restart_z].to_vec();

        let var_params = VariationalParameters::from_parts(
            pi,
            theta,
            z,
            num_mutations,
            num_clusters,
            num_samples,
            num_grid_points,
        )
        .map_err(|e| format!("invalid compat init at restart {}: {}", restart_index, e))?;
        list.push(var_params);
    }

    Ok(Some(list))
}

fn compare_restart_metrics(left: &RestartMetric, right: &RestartMetric) -> std::cmp::Ordering {
    match left.final_elbo.total_cmp(&right.final_elbo) {
        std::cmp::Ordering::Equal => right.restart_index.cmp(&left.restart_index),
        other => other,
    }
}

pub(crate) fn best_restart_index(outcomes: &[RestartOutcome]) -> Option<usize> {
    outcomes
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| compare_restart_metrics(&left.metric, &right.metric))
        .map(|(idx, _)| idx)
}

pub(crate) fn build_result_from_variational(
    var_params: &VariationalParameters,
    ccf_grid: &[f64],
    num_mutations: usize,
    num_samples: usize,
) -> Result<PcvResult, String> {
    let k = var_params.num_clusters;
    let d = var_params.num_dims;
    let g = var_params.num_grid_points;

    if d != num_samples {
        return Err("variational num_dims must equal num_samples".to_string());
    }
    if var_params.num_data_points != num_mutations {
        return Err("variational num_data_points must equal num_mutations".to_string());
    }
    if ccf_grid.len() != g {
        return Err("ccf_grid length must equal num_grid_points".to_string());
    }

    let output_grid = get_output_grid(g)?;

    let mut raw_labels = vec![0usize; num_mutations];
    let mut mutation_cluster_probs = vec![0.0; num_mutations];

    for mutation_index in 0..num_mutations {
        let row_start = mutation_index * k;
        let row = &var_params.z[row_start..row_start + k];
        let mut best_k = 0usize;
        let mut best_p = f64::NEG_INFINITY;
        for (cluster_index, value) in row.iter().enumerate() {
            if *value > best_p {
                best_p = *value;
                best_k = cluster_index;
            }
        }
        raw_labels[mutation_index] = best_k;
        mutation_cluster_probs[mutation_index] = best_p;
    }

    let mut used_clusters = BTreeSet::new();
    for label in &raw_labels {
        used_clusters.insert(*label);
    }

    let mut cluster_map = HashMap::new();
    for (new_idx, old_idx) in used_clusters.iter().enumerate() {
        cluster_map.insert(*old_idx, new_idx);
    }

    let mut mutation_cluster_ids = vec![0_i32; num_mutations];
    let mut mutation_sample_prevalence = vec![0.0; num_mutations * d];
    let mut mutation_sample_prevalence_std = vec![0.0; num_mutations * d];
    for mutation_index in 0..num_mutations {
        let new_idx = *cluster_map
            .get(&raw_labels[mutation_index])
            .ok_or_else(|| "cluster remapping failed".to_string())?;
        mutation_cluster_ids[mutation_index] = new_idx as i32;
    }

    let used_k = used_clusters.len();
    let mut cluster_sample_prevalence = vec![0.0; used_k * d];
    let mut cluster_sample_prevalence_std = vec![0.0; used_k * d];
    for cluster_index in used_clusters {
        let new_cluster_index = *cluster_map
            .get(&cluster_index)
            .ok_or_else(|| "cluster remapping failed".to_string())?;
        for dim_index in 0..d {
            let base = (cluster_index * d + dim_index) * g;
            let theta_slice = &var_params.theta[base..base + g];

            let mut mean = 0.0;
            for grid_index in 0..g {
                mean += theta_slice[grid_index] * output_grid[grid_index];
            }
            let mut variance = 0.0;
            for grid_index in 0..g {
                let diff = output_grid[grid_index] - mean;
                variance += theta_slice[grid_index] * diff * diff;
            }
            let out_idx = new_cluster_index * d + dim_index;
            cluster_sample_prevalence[out_idx] = mean;
            cluster_sample_prevalence_std[out_idx] = variance.max(0.0).sqrt();
        }
    }

    for (mutation_index, &cluster_index_raw) in
        mutation_cluster_ids.iter().enumerate().take(num_mutations)
    {
        let cluster_index = cluster_index_raw as usize;
        let cluster_offset = cluster_index * d;
        let mutation_offset = mutation_index * d;
        mutation_sample_prevalence[mutation_offset..mutation_offset + d]
            .copy_from_slice(&cluster_sample_prevalence[cluster_offset..cluster_offset + d]);
        mutation_sample_prevalence_std[mutation_offset..mutation_offset + d]
            .copy_from_slice(&cluster_sample_prevalence_std[cluster_offset..cluster_offset + d]);
    }

    Ok(PcvResult {
        num_mutations,
        num_samples,
        num_clusters: used_k,
        num_saved_trace_samples: 0,
        mutation_cluster_ids,
        mutation_cluster_probs,
        mutation_sample_prevalence,
        mutation_sample_prevalence_std,
        saved_mutation_sample_prevalence: Vec::new(),
        saved_precision_trace: Vec::new(),
        cluster_sample_prevalence,
        cluster_sample_prevalence_std,
    })
}
