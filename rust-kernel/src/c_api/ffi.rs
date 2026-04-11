// FFI accessor functions intentionally dereference raw pointers passed from C callers.
// It is the C caller's responsibility to pass valid, non-null pointers.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{c_char, c_int, CStr, CString};
use std::ptr;

use rand::{RngExt, SeedableRng};

use crate::abi::{PcvConfig, PcvError, PcvResult, PcvRow};
use crate::entrypoints::{run_vi_request, BorrowedCall, VariationalCompatInit};
use crate::phyclone::compat::data::CompatDataPoint;
use crate::phyclone::compat::loader::{
    build_compat_data_points_with_rng, CompatClusterRow, CompatInputRow, CompatLoaderConfig,
};
use crate::phyclone::compat::smc::ProposalFamily;
use crate::phyclone::compat::trace::{
    compat_trace_records_to_jsonl, run_compat_mcmc_traces_from_data_points,
    run_phyclone_mcmc_traces_from_data_points, CompatClusterSummary, CompatSampleProfile,
};
use crate::types::Density;

/// Options sent from Crystal as part of a [`CompatRunRequest`].
/// All fields are optional; defaults mirror [`CompatLoaderConfig`] / [`PhyCloneRunConfig`].
#[derive(serde::Deserialize, Default)]
struct CompatRunOptions {
    /// 0 = Binomial, 1 (or absent) = BetaBinomial.
    #[serde(default)]
    density_code: Option<u8>,
    #[serde(default)]
    precision: Option<f64>,
    #[serde(default)]
    grid_size: Option<usize>,
    #[serde(default)]
    outlier_prob: Option<f64>,
    #[serde(default)]
    num_particles: Option<usize>,
    #[serde(default)]
    burnin: Option<usize>,
    #[serde(default)]
    max_time: Option<f64>,
    #[serde(default)]
    print_freq: Option<usize>,
    #[serde(default)]
    thin: Option<usize>,
    #[serde(default)]
    resample_threshold: Option<f64>,
    #[serde(default)]
    use_phyclone_mcmc: Option<bool>,
    #[serde(default)]
    proposal_code: Option<u8>,
    #[serde(default)]
    num_samples_data_point: Option<usize>,
    #[serde(default)]
    num_samples_prune_regraft: Option<usize>,
    #[serde(default)]
    subtree_update_prob: Option<f64>,
    #[serde(default)]
    concentration_update: Option<bool>,
    #[serde(default)]
    concentration_value: Option<f64>,
    #[serde(default)]
    assign_loss_prob: Option<bool>,
    #[serde(default)]
    user_provided_loss_prob: Option<bool>,
    #[serde(default)]
    loss_prob: Option<f64>,
    #[serde(default)]
    high_loss_prob: Option<f64>,
}

/// Raw-row request format sent by Crystal.
#[derive(serde::Deserialize)]
struct CompatRunRequest {
    rows: Vec<CompatInputRow>,
    #[serde(default)]
    cluster_rows: Vec<CompatClusterRow>,
    #[serde(default)]
    options: CompatRunOptions,
}

#[cfg(test)]
mod tests {
    use super::{
        pcv_error_free, pcv_error_message, pcv_fit, pcv_phyclone_generate_trace,
        pcv_result_cluster_sample_prevalence, pcv_result_cluster_sample_prevalence_std,
        pcv_result_free, pcv_result_mutation_cluster_ids, pcv_result_mutation_cluster_probs,
        pcv_result_num_clusters, pcv_result_num_mutations, pcv_result_num_samples, pcv_string_free,
    };
    use crate::abi::{PcvConfig, PcvError, PcvResult, PcvRow};
    use crate::entrypoints::{
        best_restart_index, build_result_from_variational, get_output_grid, RestartMetric,
        RestartOutcome,
    };
    use crate::types::VariationalParameters;
    use std::ffi::{c_char, CStr, CString};
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

    #[test]
    fn pcv_phyclone_generate_trace_object_request_uses_compat_path() {
        let cluster_json = CString::new(
            r#"{"rows":[{"mutation_id":"m0","sample_id":"s0","ref_counts":30,"alt_counts":10,"major_cn":1,"minor_cn":1,"normal_cn":2,"tumour_content":1.0,"error_rate":0.001,"cluster_id":null,"chrom":"1","loss_prob":0.2,"outlier_prob":0.01},{"mutation_id":"m0","sample_id":"s1","ref_counts":25,"alt_counts":8,"major_cn":1,"minor_cn":1,"normal_cn":2,"tumour_content":1.0,"error_rate":0.001,"cluster_id":null,"chrom":"1","loss_prob":0.2,"outlier_prob":0.01}],"cluster_rows":[{"mutation_id":"m0","cluster_id":"cloneA","cellular_prevalence":0.35,"outlier_prob":0.02}],"options":{"density_code":1,"precision":200.0,"grid_size":21,"outlier_prob":0.001,"burnin":0}}"#,
        )
        .unwrap();
        let mut out_json: *mut c_char = ptr::null_mut();
        let mut out_error: *mut PcvError = ptr::null_mut();

        let rc = pcv_phyclone_generate_trace(
            cluster_json.as_ptr(),
            1,
            6,
            1,
            17,
            &mut out_json,
            &mut out_error,
        );

        assert_eq!(rc, 0);
        assert!(out_error.is_null());
        assert!(!out_json.is_null());

        let jsonl = unsafe { CStr::from_ptr(out_json) }.to_str().unwrap();
        let docs: Vec<serde_json::Value> = jsonl
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(docs.len(), 6);
        assert!(docs
            .iter()
            .all(|doc| doc["schema_version"].as_i64() == Some(1)));
        assert!(docs.iter().all(|doc| {
            doc["topology_id"]
                .as_str()
                .map(|id| id.starts_with("smc-") || id.starts_with("compat-"))
                .unwrap_or(false)
        }));
        assert!(docs
            .iter()
            .all(|doc| doc["outlier_assignments"].as_array().map(Vec::len) == Some(1)));
        assert!(docs
            .iter()
            .all(|doc| doc["clusters"].as_array().map(Vec::len) == Some(1)));

        pcv_string_free(out_json);
    }

    #[test]
    fn pcv_phyclone_generate_trace_succeeds_with_assign_loss_prob() {
        // Phase L2: assign_loss_prob is now supported and should succeed (with cluster_rows).
        let cluster_json = CString::new(
            r#"{"rows":[{"mutation_id":"m0","sample_id":"s0","ref_counts":30,"alt_counts":10,"major_cn":1,"minor_cn":1,"normal_cn":2,"tumour_content":1.0,"error_rate":0.001,"cluster_id":null,"chrom":"1","loss_prob":0.2,"outlier_prob":0.01},{"mutation_id":"m0","sample_id":"s1","ref_counts":25,"alt_counts":8,"major_cn":1,"minor_cn":1,"normal_cn":2,"tumour_content":1.0,"error_rate":0.001,"cluster_id":null,"chrom":"1","loss_prob":0.2,"outlier_prob":0.01}],"cluster_rows":[{"mutation_id":"m0","cluster_id":"cloneA","cellular_prevalence":0.35,"outlier_prob":0.02}],"options":{"density_code":1,"precision":200.0,"grid_size":21,"outlier_prob":0.001,"burnin":0,"assign_loss_prob":true}}"#,
        )
        .unwrap();
        let mut out_json: *mut c_char = ptr::null_mut();
        let mut out_error: *mut PcvError = ptr::null_mut();

        let rc = pcv_phyclone_generate_trace(
            cluster_json.as_ptr(),
            1,
            6,
            1,
            17,
            &mut out_json,
            &mut out_error,
        );

        // Should succeed (rc=0).
        assert_eq!(rc, 0);
        assert!(!out_json.is_null());
        assert!(out_error.is_null());
        pcv_string_free(out_json);
    }

    #[test]
    fn pcv_phyclone_generate_trace_rejects_loss_prob_modes_without_cluster_file() {
        // Both assign_loss_prob and user_provided_loss_prob require --cluster-file.
        let cluster_json = CString::new(
            r#"{"rows":[{"mutation_id":"m0","sample_id":"s0","ref_counts":30,"alt_counts":10,"major_cn":1,"minor_cn":1,"normal_cn":2,"tumour_content":1.0,"error_rate":0.001,"cluster_id":null,"chrom":"1","loss_prob":0.2,"outlier_prob":0.01},{"mutation_id":"m0","sample_id":"s1","ref_counts":25,"alt_counts":8,"major_cn":1,"minor_cn":1,"normal_cn":2,"tumour_content":1.0,"error_rate":0.001,"cluster_id":null,"chrom":"1","loss_prob":0.2,"outlier_prob":0.01}],"cluster_rows":[],"options":{"density_code":1,"precision":200.0,"grid_size":21,"outlier_prob":0.001,"burnin":0,"assign_loss_prob":true}}"#,
        )
        .unwrap();
        let mut out_json: *mut c_char = ptr::null_mut();
        let mut out_error: *mut PcvError = ptr::null_mut();

        let rc = pcv_phyclone_generate_trace(
            cluster_json.as_ptr(),
            1,
            6,
            1,
            17,
            &mut out_json,
            &mut out_error,
        );

        assert_eq!(rc, 1);
        assert!(out_json.is_null());
        assert!(!out_error.is_null());
        let msg = unsafe { CStr::from_ptr(pcv_error_message(out_error)) }
            .to_str()
            .unwrap();
        assert!(msg.contains("require --cluster-file"));
        pcv_error_free(out_error);
    }

    #[test]
    fn pcv_phyclone_generate_trace_rejects_unsupported_bare_array_request() {
        let cluster_json =
            CString::new(r#"[{"cluster_id":0,"mutation_ids":["m0"],"sample_ids":["s0"]}]"#)
                .unwrap();
        let mut out_json: *mut c_char = ptr::null_mut();
        let mut out_error: *mut PcvError = ptr::null_mut();

        let rc = pcv_phyclone_generate_trace(
            cluster_json.as_ptr(),
            1,
            2,
            1,
            17,
            &mut out_json,
            &mut out_error,
        );

        assert_eq!(rc, 1);
        assert!(out_json.is_null());
        assert!(!out_error.is_null());

        let msg = unsafe { CStr::from_ptr(pcv_error_message(out_error)) }
            .to_str()
            .unwrap();
        assert!(msg.contains("expected object with 'clusters'"));
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

fn assign_error(out_error: *mut *mut PcvError, message: &str) {
    if out_error.is_null() {
        return;
    }

    unsafe {
        *out_error = make_error(message);
    }
}

fn run_ffi_entrypoint<F>(
    out_result: *mut *mut PcvResult,
    out_error: *mut *mut PcvError,
    body: F,
) -> c_int
where
    F: FnOnce() -> Result<PcvResult, String>,
{
    if !out_error.is_null() {
        unsafe {
            *out_error = ptr::null_mut();
        }
    }
    if out_result.is_null() {
        assign_error(out_error, "out_result is null");
        return 1;
    }

    match body() {
        Ok(result) => {
            unsafe {
                *out_result = Box::into_raw(Box::new(result));
            }
            0
        }
        Err(message) => {
            assign_error(out_error, &message);
            1
        }
    }
}

fn run_string_entrypoint<F>(
    out_json: *mut *mut c_char,
    out_error: *mut *mut PcvError,
    body: F,
) -> c_int
where
    F: FnOnce() -> Result<String, String>,
{
    if !out_error.is_null() {
        unsafe {
            *out_error = ptr::null_mut();
        }
    }
    if out_json.is_null() {
        assign_error(out_error, "out_json is null");
        return 1;
    }

    match body() {
        Ok(json) => match CString::new(json) {
            Ok(c_string) => {
                unsafe {
                    *out_json = c_string.into_raw();
                }
                0
            }
            Err(_) => {
                assign_error(out_error, "generated JSON contains an interior null byte");
                1
            }
        },
        Err(message) => {
            assign_error(out_error, &message);
            1
        }
    }
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
    pcv_fit_with_init(
        config,
        rows,
        rows_len,
        num_mutations,
        num_samples,
        ptr::null(),
        0,
        ptr::null(),
        0,
        ptr::null(),
        0,
        out_result,
        out_error,
    )
}

#[no_mangle]
pub extern "C" fn pcv_fit_with_init(
    config: *const PcvConfig,
    rows: *const PcvRow,
    rows_len: usize,
    num_mutations: usize,
    num_samples: usize,
    compat_pi: *const f64,
    compat_pi_len: usize,
    compat_theta: *const f64,
    compat_theta_len: usize,
    compat_z: *const f64,
    compat_z_len: usize,
    out_result: *mut *mut PcvResult,
    out_error: *mut *mut PcvError,
) -> c_int {
    run_ffi_entrypoint(out_result, out_error, || {
        let request =
            unsafe { BorrowedCall::borrow(config, rows, rows_len, num_mutations, num_samples)? };
        run_vi_request(
            request,
            VariationalCompatInit {
                compat_pi,
                compat_pi_len,
                compat_theta,
                compat_theta_len,
                compat_z,
                compat_z_len,
            },
        )
    })
}

#[no_mangle]
pub extern "C" fn pcv_phyclone_generate_trace(
    cluster_json: *const c_char,
    num_chains: i32,
    num_iters: i32,
    use_seed: u8,
    seed: u64,
    out_json: *mut *mut c_char,
    out_error: *mut *mut PcvError,
) -> c_int {
    run_string_entrypoint(out_json, out_error, || {
        if cluster_json.is_null() {
            return Err("cluster_json is null".to_string());
        }
        if num_chains <= 0 {
            return Err("num_chains must be > 0".to_string());
        }
        if num_iters <= 0 {
            return Err("num_iters must be > 0".to_string());
        }

        let cluster_json = unsafe { CStr::from_ptr(cluster_json) }
            .to_str()
            .map_err(|_| "cluster_json is not valid UTF-8".to_string())?;

        let effective_seed = if use_seed == 0 { None } else { Some(seed) };

        let run_req = serde_json::from_str::<CompatRunRequest>(cluster_json).map_err(|err| {
            if cluster_json.trim_start().starts_with('[') {
                "expected object with 'clusters' key; got bare array (legacy format not supported)"
                    .to_string()
            } else {
                format!("invalid phy cluster json: {err}")
            }
        })?;

        // Validation for assign_loss_prob and user_provided_loss_prob.
        let assign_loss_prob = run_req.options.assign_loss_prob.unwrap_or(false);
        let user_provided_loss_prob = run_req.options.user_provided_loss_prob.unwrap_or(false);

        if assign_loss_prob && user_provided_loss_prob {
            return Err(
                "--assign-loss-prob and --user-provided-loss-prob are mutually exclusive"
                    .to_string(),
            );
        }

        if (assign_loss_prob || user_provided_loss_prob) && run_req.cluster_rows.is_empty() {
            return Err(
                "--assign-loss-prob and --user-provided-loss-prob require --cluster-file"
                    .to_string(),
            );
        }

        let loader_config = CompatLoaderConfig {
            density: match run_req.options.density_code {
                Some(0) => Density::Binomial,
                _ => Density::BetaBinomial,
            },
            precision: run_req.options.precision.unwrap_or(400.0),
            grid_size: run_req.options.grid_size.unwrap_or(101),
            global_outlier_prob: run_req.options.outlier_prob.unwrap_or(0.0),
            assign_loss_prob: run_req.options.assign_loss_prob.unwrap_or(false),
            user_provided_loss_prob: run_req.options.user_provided_loss_prob.unwrap_or(false),
            loss_prob: run_req.options.loss_prob.unwrap_or(0.0),
            high_loss_prob: run_req.options.high_loss_prob.unwrap_or(0.4),
        };
        let cluster_rows_opt = if run_req.cluster_rows.is_empty() {
            None
        } else {
            Some(run_req.cluster_rows.as_slice())
        };

        let mut shared_rng = match effective_seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::seed_from_u64(rand::random::<u64>()),
        };

        let data_points = build_compat_data_points_with_rng(
            &run_req.rows,
            cluster_rows_opt,
            &loader_config,
            &mut shared_rng,
        )
        .map_err(|e| format!("compat loader: {e}"))?;

        // PhyClone parity direction: loss-probability RNG consumes from the same
        // seeded stream before MCMC chain seeding.
        // Keep the serialized seed JSON-compatible with Crystal's Int64 parser.
        const JSON_SAFE_I64_MAX_U64: u64 = i64::MAX as u64;
        let mcmc_seed = effective_seed.map(|_| shared_rng.random::<u64>() & JSON_SAFE_I64_MAX_U64);

        let trace_clusters =
            build_trace_cluster_summaries_from_data_points(&data_points, &run_req.rows);

        let num_particles = run_req.options.num_particles.unwrap_or(100).max(2);
        let burnin = run_req.options.burnin.unwrap_or(1000);
        let max_time = run_req.options.max_time.unwrap_or(f64::INFINITY);
        if !max_time.is_finite() && !max_time.is_infinite() {
            return Err(format!(
                "max_time must be finite and >= 0, or Infinity (got {})",
                max_time
            ));
        }
        if max_time.is_finite() && max_time < 0.0 {
            return Err(format!("max_time must be >= 0 (got {})", max_time));
        }
        let print_freq = run_req.options.print_freq.unwrap_or(100).max(1);
        let thin = run_req.options.thin.unwrap_or(1).max(1);
        let resample_threshold = run_req
            .options
            .resample_threshold
            .unwrap_or(0.5)
            .clamp(0.0, 1.0);
        let proposal = ProposalFamily::from_code(run_req.options.proposal_code.unwrap_or(2));
        let num_samples_data_point = run_req.options.num_samples_data_point.unwrap_or(1);
        let num_samples_prune_regraft = run_req.options.num_samples_prune_regraft.unwrap_or(1);
        let subtree_update_prob = run_req.options.subtree_update_prob.unwrap_or(0.0);
        if !(0.0..=1.0).contains(&subtree_update_prob) {
            return Err(format!(
                "subtree_update_prob must be in [0, 1] (got {})",
                subtree_update_prob
            ));
        }
        let concentration_update = run_req.options.concentration_update.unwrap_or(true);
        let concentration_value = run_req.options.concentration_value.unwrap_or(1.0);
        if concentration_value <= 0.0 || !concentration_value.is_finite() {
            return Err(format!(
                "concentration_value must be finite and > 0 (got {})",
                concentration_value
            ));
        }

        let records = if run_req.options.use_phyclone_mcmc.unwrap_or(false) {
            run_phyclone_mcmc_traces_from_data_points(
                num_chains,
                num_iters,
                burnin,
                thin,
                &data_points,
                &trace_clusters,
                mcmc_seed,
                num_particles,
                resample_threshold,
                max_time,
                print_freq,
                proposal,
                num_samples_data_point,
                num_samples_prune_regraft,
                concentration_update,
                concentration_value,
                subtree_update_prob,
            )?
        } else {
            run_compat_mcmc_traces_from_data_points(
                num_chains,
                num_iters,
                burnin,
                thin,
                &data_points,
                &trace_clusters,
                mcmc_seed,
                num_particles,
                resample_threshold,
            )
        };
        compat_trace_records_to_jsonl(&records)
            .map_err(|err| format!("failed to serialize phy trace: {err}"))
    })
}

/// Builds [`CompatClusterSummary`] entries from [`CompatDataPoint`] objects produced by
/// the compat loader. Only mutations that passed PhyClone-style filtering are included,
/// so the smoke-trace cluster list is consistent with the loader interpretation.
///
/// `rows` is used to aggregate `ref_counts`/`alt_counts` per sample for each cluster.
fn build_trace_cluster_summaries_from_data_points(
    data_points: &[CompatDataPoint],
    rows: &[CompatInputRow],
) -> Vec<CompatClusterSummary> {
    use std::collections::HashMap;

    // Build lookup: mutation_id -> (ref_counts, alt_counts) per sample_id
    let mut row_lookup: HashMap<(&str, &str), (i32, i32)> = HashMap::new();
    for row in rows {
        let entry = row_lookup
            .entry((row.mutation_id.as_str(), row.sample_id.as_str()))
            .or_insert((0, 0));
        entry.0 += row.ref_counts;
        entry.1 += row.alt_counts;
    }

    data_points
        .iter()
        .map(|dp| {
            let sample_profiles: Vec<CompatSampleProfile> = dp
                .sample_ids
                .iter()
                .map(|sid| {
                    // Aggregate ref/alt across all mutations in this cluster for the sample
                    let (ref_counts, alt_counts) =
                        dp.mutation_ids.iter().fold((0i32, 0i32), |(r, a), mid| {
                            let (dr, da) = row_lookup
                                .get(&(mid.as_str(), sid.as_str()))
                                .copied()
                                .unwrap_or((0, 0));
                            (r + dr, a + da)
                        });
                    CompatSampleProfile {
                        sample_id: sid.clone(),
                        ref_counts,
                        alt_counts,
                    }
                })
                .collect();
            CompatClusterSummary {
                cluster_id: dp.idx as i32,
                mutation_ids: dp.mutation_ids.clone(),
                sample_ids: dp.sample_ids.clone(),
                sample_profiles,
                outlier_data_points: Vec::new(),
            }
        })
        .collect()
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
pub extern "C" fn pcv_result_mutation_sample_prevalence_std(
    result: *const PcvResult,
) -> *const f64 {
    if result.is_null() {
        return ptr::null();
    }
    unsafe { (*result).mutation_sample_prevalence_std.as_ptr() }
}

#[no_mangle]
pub extern "C" fn pcv_result_saved_mutation_sample_prevalence(
    result: *const PcvResult,
) -> *const f64 {
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
pub extern "C" fn pcv_string_free(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(value));
    }
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
