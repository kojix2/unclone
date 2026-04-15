mod postprocess;
mod sampler;
mod shared;
mod state;

use crate::types::{Density, McmcTrace, PcvMcmcConfig, PcvResult, PcvRow};
use postprocess::build_result_from_trace;
use rand::rngs::StdRng;
use rand::SeedableRng;
use sampler::{atom_step, concentration_step, partition_step, precision_step, save_state};
use state::{build_data_matrix, initialize_state};

pub fn fit_mcmc_model(
    cfg: &PcvMcmcConfig,
    rows: &[PcvRow],
    num_mutations: usize,
    num_samples: usize,
) -> Result<PcvResult, String> {
    if cfg.num_iters <= 0 {
        return Err("num_iters must be > 0".to_string());
    }
    if cfg.burnin < 0 {
        return Err("burnin must be >= 0".to_string());
    }
    if cfg.thin <= 0 {
        return Err("thin must be > 0".to_string());
    }
    if cfg.num_clusters <= 0 {
        return Err("num_clusters must be > 0".to_string());
    }
    if cfg.alpha <= 0.0 {
        return Err("alpha must be > 0".to_string());
    }

    let density = Density::try_from(cfg.density)?;
    let (data, observed_phi) = build_data_matrix(rows, num_mutations, num_samples)?;
    let mut rng = if cfg.use_seed == 1 {
        StdRng::seed_from_u64(cfg.seed)
    } else {
        StdRng::seed_from_u64(rand::random::<u64>())
    };
    let mut state = initialize_state(cfg, num_mutations, num_samples, &observed_phi, &mut rng)?;

    let mut trace = McmcTrace {
        co_cluster_counts: vec![0_u32; num_mutations * num_mutations],
        num_samples: 0,
        ccf_sum: vec![0.0; num_mutations * num_samples],
        ccf_sum_sq: vec![0.0; num_mutations * num_samples],
        precision_sum: 0.0,
        saved_ccf_trace: Vec::new(),
        saved_precision_trace: Vec::new(),
    };

    let total_iterations = cfg.num_iters as usize;
    let sampler_max_clusters = num_mutations.max(1);
    for iteration in 0..total_iterations {
        partition_step(
            &mut state,
            &data,
            num_mutations,
            num_samples,
            density,
            sampler_max_clusters,
            cfg.base_measure_alpha,
            cfg.base_measure_beta,
            &mut rng,
        )?;
        atom_step(
            &mut state,
            &data,
            num_samples,
            density,
            cfg.base_measure_alpha,
            cfg.base_measure_beta,
            &mut rng,
        );
        precision_step(
            &mut state,
            &data,
            num_samples,
            density,
            cfg.mh_precision_step,
            cfg.mh_precision_proposal_precision,
            &mut rng,
        );
        concentration_step(&mut state, num_mutations, cfg, &mut rng)?;

        if cfg.print_freq > 0 && (iteration + 1) % cfg.print_freq as usize == 0 {
            eprintln!(
                "[tyclone:mcmc] iter={} clusters={} alpha={:.4} precision={:.4}",
                iteration + 1,
                state.atoms.len(),
                state.alpha,
                state.precision
            );
        }

        if iteration >= cfg.burnin as usize && ((iteration - cfg.burnin as usize) % cfg.thin as usize == 0) {
            save_state(&mut trace, &state, num_mutations, num_samples);
        }
    }

    build_result_from_trace(
        &trace,
        &data,
        density,
        num_mutations,
        num_samples,
        cfg.num_clusters as usize,
    )
}

#[cfg(test)]
mod tests {
    use super::fit_mcmc_model;
    use super::postprocess::{cluster_with_mpear, compute_mpear};
    use crate::types::PcvMcmcConfig;
    use crate::types::PcvRow;

    fn default_mcmc_config() -> PcvMcmcConfig {
        PcvMcmcConfig {
            num_iters: 20,
            burnin: 10,
            thin: 2,
            num_clusters: 4,
            alpha: 1.0,
            alpha_prior_shape: 1.0,
            alpha_prior_rate: 0.001,
            init_method: 0,
            base_measure_alpha: 1.0,
            base_measure_beta: 1.0,
            mh_step_size: 0.05,
            mh_precision_step: 0.0,
            mh_precision_proposal_precision: 0.01,
            precision: 1000.0,
            density: 0,
            use_seed: 1,
            seed: 7,
            print_freq: 0,
        }
    }

    #[test]
    fn fit_mcmc_model_returns_well_formed_result() {
        let rows = vec![
            PcvRow {
                mutation_index: 0,
                sample_index: 0,
                ref_counts: 30,
                alt_counts: 10,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.8,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 0,
                sample_index: 1,
                ref_counts: 25,
                alt_counts: 12,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.85,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 1,
                sample_index: 0,
                ref_counts: 15,
                alt_counts: 20,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.8,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 1,
                sample_index: 1,
                ref_counts: 10,
                alt_counts: 22,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.85,
                error_rate: 1e-3,
            },
        ];

        let result = fit_mcmc_model(&default_mcmc_config(), &rows, 2, 2).unwrap();
        assert_eq!(result.num_mutations, 2);
        assert_eq!(result.num_samples, 2);
        assert!((1..=2).contains(&result.num_clusters));
        assert_eq!(result.mutation_cluster_ids.len(), 2);
        assert_eq!(result.mutation_cluster_probs.len(), 2);
        assert_eq!(
            result.cluster_sample_prevalence.len(),
            result.num_clusters * 2
        );
        assert_eq!(
            result.cluster_sample_prevalence_std.len(),
            result.num_clusters * 2
        );
    }

    #[test]
    fn num_iters_counts_total_iterations_before_burnin_and_thinning() {
        let rows = vec![
            PcvRow {
                mutation_index: 0,
                sample_index: 0,
                ref_counts: 30,
                alt_counts: 10,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.8,
                error_rate: 1e-3,
            },
            PcvRow {
                mutation_index: 1,
                sample_index: 0,
                ref_counts: 15,
                alt_counts: 20,
                major_cn: 2,
                minor_cn: 1,
                normal_cn: 2,
                tumour_content: 0.8,
                error_rate: 1e-3,
            },
        ];

        let mut cfg = default_mcmc_config();
        cfg.num_iters = 20;
        cfg.burnin = 10;
        cfg.thin = 2;

        let result = fit_mcmc_model(&cfg, &rows, 2, 1).unwrap();
        assert_eq!(result.num_saved_trace_samples, 5);
    }

    #[test]
    fn mpear_prefers_two_block_partition_for_block_similarity_matrix() {
        let sim_mat = vec![
            1.0, 0.95, 0.10, 0.10, 0.95, 1.0, 0.10, 0.10, 0.10, 0.10, 1.0, 0.90, 0.10, 0.10, 0.90,
            1.0,
        ];

        let labels = cluster_with_mpear(&sim_mat, 4, 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }

    #[test]
    fn mpear_score_improves_over_single_cluster_for_block_structure() {
        let sim_mat = vec![
            1.0, 0.95, 0.10, 0.10, 0.95, 1.0, 0.10, 0.10, 0.10, 0.10, 1.0, 0.90, 0.10, 0.10, 0.90,
            1.0,
        ];

        let one_cluster = vec![0usize, 0, 0, 0];
        let two_clusters = vec![0usize, 0, 1, 1];

        assert!(
            compute_mpear(&two_clusters, &sim_mat, 4) > compute_mpear(&one_cluster, &sim_mat, 4)
        );
    }
}
