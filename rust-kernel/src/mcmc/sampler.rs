use crate::types::{ClusterAtom, Density, DpState, McmcTrace, PcvMcmcConfig, SampleDataPoint};
use rand::seq::SliceRandom;
use rand::{Rng, RngExt};
use rand_distr::{Beta, Distribution, Gamma};

use super::shared::{
    base_measure_log_p_atom, cluster_atom_log_posterior, cluster_members, mutation_log_likelihood,
    sample_log_weights, sample_prior_atom, total_log_likelihood, AUX_NEW_CLUSTERS, EPS,
};

const DEFAULT_PRECISION_PRIOR_SHAPE: f64 = 1.0;
const DEFAULT_PRECISION_PRIOR_RATE: f64 = 0.001;

fn log_gamma_pdf(x: f64, shape: f64, rate: f64) -> f64 {
    use statrs::function::gamma::ln_gamma;

    -ln_gamma(shape) + shape * rate.ln() + (shape - 1.0) * x.ln() - rate * x
}

fn gamma_proposal_params(current: f64, proposal_precision: f64) -> (f64, f64) {
    let rate = current * proposal_precision;
    let shape = rate * current;
    (shape.max(EPS), rate.max(EPS))
}

pub fn partition_step(
    state: &mut DpState,
    data: &[SampleDataPoint],
    num_mutations: usize,
    num_samples: usize,
    density: Density,
    max_clusters: usize,
    base_measure_alpha: f64,
    base_measure_beta: f64,
    rng: &mut impl Rng,
) -> Result<(), String> {
    let mut counts = vec![0usize; state.atoms.len()];
    for &cluster_index in &state.cluster_id {
        counts[cluster_index] += 1;
    }

    let mut mutation_order = (0..num_mutations).collect::<Vec<_>>();
    mutation_order.shuffle(rng);

    for mutation_index in mutation_order {
        let old_cluster = state.cluster_id[mutation_index];
        let old_atom = state.atoms[old_cluster].clone();
        counts[old_cluster] -= 1;
        let mut removed_empty_cluster = false;
        if counts[old_cluster] == 0 {
            removed_empty_cluster = true;
            counts.remove(old_cluster);
            state.atoms.remove(old_cluster);
            for label in &mut state.cluster_id {
                if *label > old_cluster {
                    *label -= 1;
                }
            }
        }

        enum Candidate {
            Existing(usize),
            New(ClusterAtom),
        }

        let mut candidates = Vec::new();
        let mut log_weights = Vec::new();

        for cluster_index in 0..state.atoms.len() {
            if counts[cluster_index] == 0 {
                continue;
            }
            log_weights.push(
                (counts[cluster_index] as f64).ln()
                    + mutation_log_likelihood(
                        mutation_index,
                        &state.atoms[cluster_index],
                        data,
                        num_samples,
                        density,
                        state.precision,
                    ),
            );
            candidates.push(Candidate::Existing(cluster_index));
        }

        if state.atoms.len() < max_clusters {
            if removed_empty_cluster {
                log_weights.push(
                    (state.alpha / AUX_NEW_CLUSTERS as f64).ln()
                        + mutation_log_likelihood(
                            mutation_index,
                            &old_atom,
                            data,
                            num_samples,
                            density,
                            state.precision,
                        ),
                );
                candidates.push(Candidate::New(old_atom));
            }

            let num_auxiliary = if removed_empty_cluster {
                AUX_NEW_CLUSTERS.saturating_sub(1)
            } else {
                AUX_NEW_CLUSTERS
            };

            for _ in 0..num_auxiliary {
                let atom =
                    sample_prior_atom(num_samples, base_measure_alpha, base_measure_beta, rng)?;
                log_weights.push(
                    (state.alpha / AUX_NEW_CLUSTERS as f64).ln()
                        + mutation_log_likelihood(
                            mutation_index,
                            &atom,
                            data,
                            num_samples,
                            density,
                            state.precision,
                        ),
                );
                candidates.push(Candidate::New(atom));
            }
        }

        let selected = sample_log_weights(&log_weights, rng)?;
        match candidates.swap_remove(selected) {
            Candidate::Existing(cluster_index) => {
                state.cluster_id[mutation_index] = cluster_index;
                counts[cluster_index] += 1;
            }
            Candidate::New(atom) => {
                let cluster_index = state.atoms.len();
                state.atoms.push(atom);
                counts.push(1);
                state.cluster_id[mutation_index] = cluster_index;
            }
        }
    }

    Ok(())
}

pub fn atom_step(
    state: &mut DpState,
    data: &[SampleDataPoint],
    num_samples: usize,
    density: Density,
    base_measure_alpha: f64,
    base_measure_beta: f64,
    rng: &mut impl Rng,
) {
    let members = cluster_members(state);
    for (cluster_index, member_indices) in members.iter().enumerate() {
        if member_indices.is_empty() {
            continue;
        }

        for sample_index in 0..num_samples {
            let Ok(proposal_scalar) =
                sample_prior_atom(1, base_measure_alpha, base_measure_beta, rng)
            else {
                continue;
            };

            let current_atom = state.atoms[cluster_index].clone();
            let mut proposal_atom = current_atom.clone();
            proposal_atom.phi[sample_index] = proposal_scalar.phi[0];

            let current_lp = cluster_atom_log_posterior(
                &current_atom,
                member_indices,
                data,
                num_samples,
                density,
                state.precision,
                base_measure_alpha,
                base_measure_beta,
            );
            let proposal_lp = cluster_atom_log_posterior(
                &proposal_atom,
                member_indices,
                data,
                num_samples,
                density,
                state.precision,
                base_measure_alpha,
                base_measure_beta,
            );

            let forward_log_ratio = proposal_lp
                - base_measure_log_p_atom(&proposal_atom, base_measure_alpha, base_measure_beta);
            let reverse_log_ratio = current_lp
                - base_measure_log_p_atom(&current_atom, base_measure_alpha, base_measure_beta);

            if rng.random::<f64>().ln() < (forward_log_ratio - reverse_log_ratio).min(0.0) {
                state.atoms[cluster_index] = proposal_atom;
            }
        }
    }
}

pub fn precision_step(
    state: &mut DpState,
    data: &[SampleDataPoint],
    num_samples: usize,
    density: Density,
    mh_precision_step: f64,
    proposal_precision: f64,
    rng: &mut impl Rng,
) {
    if density != Density::BetaBinomial || mh_precision_step <= 0.0 || proposal_precision <= 0.0 {
        return;
    }

    let old_precision = state.precision.max(EPS);
    let (proposal_shape, proposal_rate) = gamma_proposal_params(old_precision, proposal_precision);
    let proposal_dist = match Gamma::new(proposal_shape, 1.0 / proposal_rate) {
        Ok(dist) => dist,
        Err(_) => return,
    };
    let new_precision = proposal_dist.sample(rng).max(EPS);

    let current_ll = total_log_likelihood(state, data, num_samples, density)
        + log_gamma_pdf(
            old_precision,
            DEFAULT_PRECISION_PRIOR_SHAPE,
            DEFAULT_PRECISION_PRIOR_RATE,
        );

    let old_value = state.precision;
    state.precision = new_precision;
    let proposal_ll = total_log_likelihood(state, data, num_samples, density)
        + log_gamma_pdf(
            new_precision,
            DEFAULT_PRECISION_PRIOR_SHAPE,
            DEFAULT_PRECISION_PRIOR_RATE,
        );

    let (reverse_shape, reverse_rate) = gamma_proposal_params(new_precision, proposal_precision);
    let forward_log_ratio =
        proposal_ll - log_gamma_pdf(new_precision, proposal_shape, proposal_rate);
    let reverse_log_ratio = current_ll - log_gamma_pdf(old_precision, reverse_shape, reverse_rate);

    if rng.random::<f64>().ln() >= (forward_log_ratio - reverse_log_ratio).min(0.0) {
        state.precision = old_value;
    }
}

pub fn concentration_step(
    state: &mut DpState,
    num_mutations: usize,
    cfg: &PcvMcmcConfig,
    rng: &mut impl Rng,
) -> Result<(), String> {
    if cfg.alpha_prior_shape <= 0.0 || cfg.alpha_prior_rate <= 0.0 {
        return Err("alpha prior shape and rate must be > 0".to_string());
    }

    let clusters = state.atoms.len() as f64;
    let beta = Beta::new(state.alpha + 1.0, num_mutations as f64)
        .map_err(|error| format!("failed to initialize beta sampler: {error}"))?;
    let eta = beta.sample(rng).clamp(EPS, 1.0 - EPS);
    let mix = (cfg.alpha_prior_shape + clusters - 1.0)
        / ((num_mutations as f64) * (cfg.alpha_prior_rate - eta.ln())
            + cfg.alpha_prior_shape
            + clusters
            - 1.0);
    let shape = if rng.random::<f64>() < mix {
        cfg.alpha_prior_shape + clusters
    } else {
        cfg.alpha_prior_shape + clusters - 1.0
    };
    let rate = cfg.alpha_prior_rate - eta.ln();
    let gamma = Gamma::new(shape, 1.0 / rate)
        .map_err(|error| format!("failed to initialize gamma sampler: {error}"))?;
    state.alpha = gamma.sample(rng).max(EPS);
    Ok(())
}

pub fn save_state(
    trace: &mut McmcTrace,
    state: &DpState,
    num_mutations: usize,
    num_samples: usize,
) {
    trace.num_samples += 1;
    trace.precision_sum += state.precision;
    trace.saved_precision_trace.push(state.precision);

    for left in 0..num_mutations {
        for right in left..num_mutations {
            if state.cluster_id[left] == state.cluster_id[right] {
                trace.co_cluster_counts[left * num_mutations + right] += 1;
                if left != right {
                    trace.co_cluster_counts[right * num_mutations + left] += 1;
                }
            }
        }

        let atom = &state.atoms[state.cluster_id[left]];
        for sample_index in 0..num_samples {
            let value = atom.phi[sample_index];
            let offset = left * num_samples + sample_index;
            trace.ccf_sum[offset] += value;
            trace.ccf_sum_sq[offset] += value * value;
            trace.saved_ccf_trace.push(value);
        }
    }
}
