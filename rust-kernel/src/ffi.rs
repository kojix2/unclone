use std::collections::{BTreeSet, HashMap};
use std::ffi::{c_char, c_int, CString};
use std::ptr;

use crate::inference::fit_variational_model;
use crate::mcmc::fit_mcmc_model;
use crate::preprocess::{build_log_p_data, build_log_p_data_parallel, get_ccf_grid};
use crate::types::{
    DataPreprocessor, Density, PcvConfig, PcvError, PcvMcmcConfig, PcvResult, PcvRow, Priors,
    VariationalParameters,
};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

fn get_output_grid(grid_size: usize) -> Result<Vec<f64>, String> {
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
struct RestartMetric {
    restart_index: usize,
    restart_seed: u64,
    final_elbo: f64,
    used_clusters: usize,
}

struct RestartOutcome {
    metric: RestartMetric,
    var_params: VariationalParameters,
}

fn restart_seed(base_seed: u64, restart_index: usize) -> u64 {
    base_seed ^ ((restart_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

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

fn compare_restart_metrics(left: &RestartMetric, right: &RestartMetric) -> std::cmp::Ordering {
    match left.final_elbo.total_cmp(&right.final_elbo) {
        std::cmp::Ordering::Equal => right.restart_index.cmp(&left.restart_index),
        other => other,
    }
}

fn best_restart_index(outcomes: &[RestartOutcome]) -> Option<usize> {
    outcomes
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| compare_restart_metrics(&left.metric, &right.metric))
        .map(|(idx, _)| idx)
}

fn build_result_from_variational(
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

    for mutation_index in 0..num_mutations {
        let cluster_index = mutation_cluster_ids[mutation_index] as usize;
        let cluster_offset = cluster_index * d;
        let mutation_offset = mutation_index * d;
        for dim_index in 0..d {
            mutation_sample_prevalence[mutation_offset + dim_index] =
                cluster_sample_prevalence[cluster_offset + dim_index];
            mutation_sample_prevalence_std[mutation_offset + dim_index] =
                cluster_sample_prevalence_std[cluster_offset + dim_index];
        }
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

#[cfg(test)]
mod tests {
    use super::{
        best_restart_index, build_result_from_variational, get_output_grid, pcv_error_free,
        pcv_error_message, pcv_fit, pcv_result_cluster_sample_prevalence,
        pcv_result_cluster_sample_prevalence_std, pcv_result_free, pcv_result_mutation_cluster_ids,
        pcv_result_mutation_cluster_probs, pcv_result_num_clusters, pcv_result_num_mutations,
        pcv_result_num_samples, RestartMetric, RestartOutcome,
    };
    use crate::types::{PcvConfig, PcvError, PcvResult, PcvRow, VariationalParameters};
    use std::ffi::CStr;
    use std::ptr;

    fn default_config() -> PcvConfig {
        PcvConfig {
            num_clusters: 2,
            num_grid_points: 8,
            num_restarts: 1,
            max_iters: 50,
            print_freq: 0,
            kernel_threads: 1,
            restart_parallelism: 1,
            convergence_threshold: 1e-6,
            mix_weight_prior: 1.0,
            precision: 1000.0,
            density: 1,
            use_seed: 1,
            seed: 7,
        }
    }

    fn default_row() -> PcvRow {
        PcvRow {
            mutation_index: 0,
            sample_index: 0,
            ref_counts: 10,
            alt_counts: 5,
            major_cn: 1,
            minor_cn: 1,
            normal_cn: 2,
            tumour_content: 1.0,
            error_rate: 0.001,
        }
    }

    #[test]
    fn renumbers_used_clusters_sequentially() {
        let var_params = VariationalParameters::from_parts(
            vec![1.0, 1.0, 1.0],
            vec![0.9, 0.1, 0.8, 0.2, 0.6, 0.4],
            vec![0.1, 0.8, 0.1, 0.2, 0.7, 0.1],
            2,
            3,
            1,
            2,
        )
        .unwrap();

        let result = build_result_from_variational(&var_params, &[0.0, 1.0], 2, 1).unwrap();
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.mutation_cluster_ids, vec![0, 0]);
        assert_eq!(result.cluster_sample_prevalence.len(), 1);
        assert_eq!(result.cluster_sample_prevalence_std.len(), 1);
    }

    #[test]
    fn output_grid_matches_pyclone_postprocess_linspace() {
        let grid = get_output_grid(5).unwrap();

        assert_eq!(grid, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn result_prevalence_uses_output_grid_not_inference_grid() {
        let var_params = VariationalParameters::from_parts(
            vec![1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0],
            1,
            1,
            1,
            3,
        )
        .unwrap();

        let result =
            build_result_from_variational(&var_params, &[1e-6, 0.5, 1.0 - 1e-6], 1, 1).unwrap();
        assert_eq!(result.cluster_sample_prevalence, vec![0.0]);
        assert_eq!(result.cluster_sample_prevalence_std, vec![0.0]);
    }

    #[test]
    fn result_accessors_handle_null_pointers() {
        assert_eq!(pcv_result_num_mutations(ptr::null()), 0);
        assert_eq!(pcv_result_num_samples(ptr::null()), 0);
        assert_eq!(pcv_result_num_clusters(ptr::null()), 0);
        assert!(pcv_result_mutation_cluster_ids(ptr::null()).is_null());
        assert!(pcv_result_mutation_cluster_probs(ptr::null()).is_null());
        assert!(pcv_result_cluster_sample_prevalence(ptr::null()).is_null());
        assert!(pcv_result_cluster_sample_prevalence_std(ptr::null()).is_null());
        pcv_result_free(ptr::null_mut());
        pcv_error_free(ptr::null_mut());
        assert!(pcv_error_message(ptr::null()).is_null());
    }

    #[test]
    fn best_restart_prefers_higher_elbo_then_lower_restart_index() {
        let outcomes = vec![
            RestartOutcome {
                metric: RestartMetric {
                    restart_index: 0,
                    restart_seed: 7,
                    final_elbo: -10.0,
                    used_clusters: 2,
                },
                var_params: VariationalParameters::from_parts(
                    vec![1.0],
                    vec![1.0, 0.0],
                    vec![1.0],
                    1,
                    1,
                    1,
                    2,
                )
                .unwrap(),
            },
            RestartOutcome {
                metric: RestartMetric {
                    restart_index: 1,
                    restart_seed: 8,
                    final_elbo: -10.0,
                    used_clusters: 2,
                },
                var_params: VariationalParameters::from_parts(
                    vec![1.0],
                    vec![1.0, 0.0],
                    vec![1.0],
                    1,
                    1,
                    1,
                    2,
                )
                .unwrap(),
            },
            RestartOutcome {
                metric: RestartMetric {
                    restart_index: 2,
                    restart_seed: 9,
                    final_elbo: -9.0,
                    used_clusters: 2,
                },
                var_params: VariationalParameters::from_parts(
                    vec![1.0],
                    vec![1.0, 0.0],
                    vec![1.0],
                    1,
                    1,
                    1,
                    2,
                )
                .unwrap(),
            },
        ];

        assert_eq!(best_restart_index(&outcomes), Some(2));
    }

    #[test]
    fn pcv_fit_returns_error_when_out_result_is_null() {
        let cfg = default_config();
        let row = default_row();
        let mut out_error: *mut PcvError = ptr::null_mut();

        let rc = pcv_fit(&cfg, &row, 1, 1, 1, ptr::null_mut(), &mut out_error);

        assert_eq!(rc, 1);
        assert!(!out_error.is_null());

        let message_ptr = pcv_error_message(out_error);
        assert!(!message_ptr.is_null());
        let message = unsafe { CStr::from_ptr(message_ptr).to_str().unwrap() };
        assert_eq!(message, "out_result is null");

        pcv_error_free(out_error);
    }

    #[test]
    fn pcv_fit_returns_error_when_config_or_rows_is_null() {
        let cfg = default_config();
        let row = default_row();
        let mut out_result: *mut PcvResult = ptr::null_mut();
        let mut out_error: *mut PcvError = ptr::null_mut();

        let rc_null_config = pcv_fit(ptr::null(), &row, 1, 1, 1, &mut out_result, &mut out_error);
        assert_eq!(rc_null_config, 1);
        assert!(!out_error.is_null());
        let msg1 = unsafe {
            CStr::from_ptr(pcv_error_message(out_error))
                .to_str()
                .unwrap()
        };
        assert_eq!(msg1, "config or rows is null");
        pcv_error_free(out_error);

        out_error = ptr::null_mut();
        let rc_null_rows = pcv_fit(&cfg, ptr::null(), 1, 1, 1, &mut out_result, &mut out_error);
        assert_eq!(rc_null_rows, 1);
        assert!(!out_error.is_null());
        let msg2 = unsafe {
            CStr::from_ptr(pcv_error_message(out_error))
                .to_str()
                .unwrap()
        };
        assert_eq!(msg2, "config or rows is null");
        pcv_error_free(out_error);
    }
}

fn make_error(message: &str) -> *mut PcvError {
    let safe = message.replace('\0', " ");
    let err = PcvError {
        message: CString::new(safe).expect("CString::new failed"),
    };
    Box::into_raw(Box::new(err))
}

#[no_mangle]
pub extern "C" fn pcv_fit(
    config: *const PcvConfig,
    rows: *const PcvRow,
    rows_len: usize,
    num_mutations: usize,
    num_samples: usize,
    out_result: *mut *mut PcvResult,
    out_error: *mut *mut PcvError,
) -> c_int {
    if !out_error.is_null() {
        unsafe {
            *out_error = ptr::null_mut();
        }
    }
    if out_result.is_null() {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("out_result is null");
            }
        }
        return 1;
    }
    if config.is_null() || rows.is_null() {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("config or rows is null");
            }
        }
        return 1;
    }
    if rows_len == 0 || num_mutations == 0 || num_samples == 0 {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("rows_len, num_mutations, num_samples must be > 0");
            }
        }
        return 1;
    }

    let cfg = unsafe { &*config };
    let input_rows = unsafe { std::slice::from_raw_parts(rows, rows_len) };
    let density = match Density::try_from(cfg.density) {
        Ok(density) => density,
        Err(message) => {
            if !out_error.is_null() {
                unsafe {
                    *out_error = make_error(&message);
                }
            }
            return 1;
        }
    };

    if cfg.num_grid_points < 2 {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("num_grid_points must be >= 2");
            }
        }
        return 1;
    }
    if cfg.num_clusters <= 0 {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("num_clusters must be > 0");
            }
        }
        return 1;
    }
    if cfg.num_restarts <= 0 {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("num_restarts must be > 0");
            }
        }
        return 1;
    }
    if cfg.kernel_threads < 0 {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("kernel_threads must be >= 0");
            }
        }
        return 1;
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
        match ThreadPoolBuilder::new().num_threads(rayon_pool_threads).build() {
            Ok(pool) => Some(pool),
            Err(error) => {
                if !out_error.is_null() {
                    unsafe {
                        *out_error =
                            make_error(&format!("failed to build rayon thread pool: {error}"));
                    }
                }
                return 1;
            }
        }
    } else {
        None
    };

    let ccf_grid = match get_ccf_grid(cfg.num_grid_points as usize, 1e-6) {
        Ok(grid) => grid,
        Err(message) => {
            if !out_error.is_null() {
                unsafe {
                    *out_error = make_error(&message);
                }
            }
            return 1;
        }
    };

    let log_p_data_result = if enable_kernel_parallel {
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
        })
    } else {
        build_log_p_data(
            input_rows,
            num_mutations,
            num_samples,
            &ccf_grid,
            density,
            cfg.precision,
        )
    };

    let log_p_data = match log_p_data_result {
        Ok(log_p_data) => log_p_data,
        Err(message) => {
            if !out_error.is_null() {
                unsafe {
                    *out_error = make_error(&message);
                }
            }
            return 1;
        }
    };

    let priors = match Priors::new(
        cfg.num_clusters as usize,
        cfg.num_grid_points as usize,
        cfg.mix_weight_prior,
    ) {
        Ok(priors) => priors,
        Err(message) => {
            if !out_error.is_null() {
                unsafe {
                    *out_error = make_error(&message);
                }
            }
            return 1;
        }
    };

    let data_preproc = DataPreprocessor::new(&log_p_data, enable_kernel_parallel);

    let base_seed = if cfg.use_seed == 1 {
        cfg.seed
    } else {
        rand::random::<u64>()
    };

    let run_all_restarts = || -> Result<Vec<RestartOutcome>, String> {
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

    let mut restart_outcomes = match run_all_restarts() {
        Ok(outcomes) => outcomes,
        Err(message) => {
            if !out_error.is_null() {
                unsafe {
                    *out_error = make_error(&message);
                }
            }
            return 1;
        }
    };

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

    let best_outcome_index = match best_restart_index(&restart_outcomes) {
        Some(index) => index,
        None => {
            if !out_error.is_null() {
                unsafe {
                    *out_error = make_error("no restart produced variational parameters");
                }
            }
            return 1;
        }
    };

    let best_elbo = restart_outcomes[best_outcome_index].metric.final_elbo;
    let best_restart_index = restart_outcomes[best_outcome_index].metric.restart_index;

    if cfg.print_freq > 0 {
        eprintln!(
            "[tyclone] best_restart={} best_final_elbo={}",
            best_restart_index, best_elbo
        );
    }

    let best_var_params = restart_outcomes.swap_remove(best_outcome_index).var_params;

    let result = match build_result_from_variational(
        &best_var_params,
        &ccf_grid,
        num_mutations,
        num_samples,
    ) {
        Ok(result) => result,
        Err(message) => {
            if !out_error.is_null() {
                unsafe {
                    *out_error = make_error(&message);
                }
            }
            return 1;
        }
    };

    unsafe {
        *out_result = Box::into_raw(Box::new(result));
    }
    0
}

#[no_mangle]
pub extern "C" fn pcv_fit_mcmc(
    config: *const PcvMcmcConfig,
    rows: *const PcvRow,
    rows_len: usize,
    num_mutations: usize,
    num_samples: usize,
    out_result: *mut *mut PcvResult,
    out_error: *mut *mut PcvError,
) -> c_int {
    if !out_error.is_null() {
        unsafe {
            *out_error = ptr::null_mut();
        }
    }
    if out_result.is_null() {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("out_result is null");
            }
        }
        return 1;
    }
    if config.is_null() || rows.is_null() {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("config or rows is null");
            }
        }
        return 1;
    }
    if rows_len == 0 || num_mutations == 0 || num_samples == 0 {
        if !out_error.is_null() {
            unsafe {
                *out_error = make_error("rows_len, num_mutations, num_samples must be > 0");
            }
        }
        return 1;
    }

    let cfg = unsafe { &*config };
    let input_rows = unsafe { std::slice::from_raw_parts(rows, rows_len) };

    let result = match fit_mcmc_model(cfg, input_rows, num_mutations, num_samples) {
        Ok(result) => result,
        Err(message) => {
            if !out_error.is_null() {
                unsafe {
                    *out_error = make_error(&message);
                }
            }
            return 1;
        }
    };

    unsafe {
        *out_result = Box::into_raw(Box::new(result));
    }
    0
}

#[no_mangle]
pub extern "C" fn pcv_result_num_mutations(result: *const PcvResult) -> usize {
    if result.is_null() {
        return 0;
    }
    unsafe { (*result).num_mutations }
}

#[no_mangle]
pub extern "C" fn pcv_result_num_samples(result: *const PcvResult) -> usize {
    if result.is_null() {
        return 0;
    }
    unsafe { (*result).num_samples }
}

#[no_mangle]
pub extern "C" fn pcv_result_num_clusters(result: *const PcvResult) -> usize {
    if result.is_null() {
        return 0;
    }
    unsafe { (*result).num_clusters }
}

#[no_mangle]
pub extern "C" fn pcv_result_num_saved_trace_samples(result: *const PcvResult) -> usize {
    if result.is_null() {
        return 0;
    }
    unsafe { (*result).num_saved_trace_samples }
}

#[no_mangle]
pub extern "C" fn pcv_result_mutation_cluster_ids(result: *const PcvResult) -> *const i32 {
    if result.is_null() {
        return ptr::null();
    }
    unsafe { (*result).mutation_cluster_ids.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_result_mutation_cluster_probs(result: *const PcvResult) -> *const f64 {
    if result.is_null() {
        return ptr::null();
    }
    unsafe { (*result).mutation_cluster_probs.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_result_mutation_sample_prevalence(result: *const PcvResult) -> *const f64 {
    if result.is_null() {
        return ptr::null();
    }
    unsafe { (*result).mutation_sample_prevalence.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_result_mutation_sample_prevalence_std(result: *const PcvResult) -> *const f64 {
    if result.is_null() {
        return ptr::null();
    }
    unsafe { (*result).mutation_sample_prevalence_std.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_result_saved_mutation_sample_prevalence(result: *const PcvResult) -> *const f64 {
    if result.is_null() {
        return ptr::null();
    }
    unsafe { (*result).saved_mutation_sample_prevalence.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_result_saved_precision_trace(result: *const PcvResult) -> *const f64 {
    if result.is_null() {
        return ptr::null();
    }
    unsafe { (*result).saved_precision_trace.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_result_cluster_sample_prevalence(result: *const PcvResult) -> *const f64 {
    if result.is_null() {
        return ptr::null();
    }
    unsafe { (*result).cluster_sample_prevalence.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_result_cluster_sample_prevalence_std(result: *const PcvResult) -> *const f64 {
    if result.is_null() {
        return ptr::null();
    }
    unsafe { (*result).cluster_sample_prevalence_std.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_result_free(result: *mut PcvResult) {
    if result.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(result));
    }
}

#[no_mangle]
pub extern "C" fn pcv_error_message(err: *const PcvError) -> *const c_char {
    if err.is_null() {
        return ptr::null();
    }
    unsafe { (*err).message.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_error_free(err: *mut PcvError) {
    if err.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(err));
    }
}
