use crate::likelihood::{log_pyclone_beta_binomial_pdf, log_pyclone_binomial_pdf};
use crate::math::log_sum_exp;
use crate::types::{Density, McmcTrace, PcvResult, SampleDataPoint};
use std::collections::HashMap;

const POSTPROCESS_MESH_SIZE: usize = 101;
const MERGE_REFINEMENT_MIN_SIMILARITY: f64 = 0.80;

pub fn posterior_similarity_matrix(trace: &McmcTrace, num_mutations: usize) -> Vec<f64> {
    let denom = trace.num_samples.max(1) as f64;
    trace
        .co_cluster_counts
        .iter()
        .take(num_mutations * num_mutations)
        .map(|&count| count as f64 / denom)
        .collect()
}

fn cluster_average_distance(
    cluster_a: &[usize],
    cluster_b: &[usize],
    dist_mat: &[f64],
    num_mutations: usize,
) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for &left in cluster_a {
        for &right in cluster_b {
            total += dist_mat[left * num_mutations + right];
            count += 1;
        }
    }
    total / count.max(1) as f64
}

fn labels_from_clusters(clusters: &[Vec<usize>], num_mutations: usize) -> Vec<usize> {
    let mut labels = vec![0usize; num_mutations];
    for (cluster_index, members) in clusters.iter().enumerate() {
        for &mutation_index in members {
            labels[mutation_index] = cluster_index;
        }
    }
    labels
}

fn clusters_from_labels(cluster_labels: &[usize]) -> Vec<Vec<usize>> {
    let num_clusters = cluster_labels
        .iter()
        .copied()
        .max()
        .map(|value| value + 1)
        .unwrap_or(0);
    let mut clusters = vec![Vec::new(); num_clusters];
    for (mutation_index, &cluster_index) in cluster_labels.iter().enumerate() {
        clusters[cluster_index].push(mutation_index);
    }
    clusters
}

fn cluster_average_similarity(
    cluster_a: &[usize],
    cluster_b: &[usize],
    sim_mat: &[f64],
    num_mutations: usize,
) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for &left in cluster_a {
        for &right in cluster_b {
            total += sim_mat[left * num_mutations + right];
            count += 1;
        }
    }
    total / count.max(1) as f64
}

fn average_linkage_labels(
    dist_mat: &[f64],
    num_mutations: usize,
    max_clusters: usize,
) -> Vec<Vec<usize>> {
    let mut clusters: Vec<Vec<usize>> = (0..num_mutations).map(|index| vec![index]).collect();
    let mut labels_by_k = vec![Vec::new(); max_clusters + 1];

    if num_mutations <= max_clusters {
        labels_by_k[num_mutations] = (0..num_mutations).collect();
    }

    while clusters.len() > 1 {
        let mut best_pair = (0usize, 1usize);
        let mut best_distance = f64::INFINITY;

        for left in 0..clusters.len() {
            for right in (left + 1)..clusters.len() {
                let distance = cluster_average_distance(
                    &clusters[left],
                    &clusters[right],
                    dist_mat,
                    num_mutations,
                );
                if distance < best_distance {
                    best_distance = distance;
                    best_pair = (left, right);
                }
            }
        }

        let (left, right) = best_pair;
        let mut merged = clusters.remove(right);
        clusters[left].append(&mut merged);

        if clusters.len() <= max_clusters {
            labels_by_k[clusters.len()] = labels_from_clusters(&clusters, num_mutations);
        }
    }

    labels_by_k[1] = vec![0usize; num_mutations];
    labels_by_k
}

pub fn compute_mpear(cluster_labels: &[usize], sim_mat: &[f64], num_mutations: usize) -> f64 {
    if num_mutations < 2 {
        return 0.0;
    }

    let mut indicator_similarity = 0.0;
    let mut indicator_total = 0.0;
    let mut similarity_total = 0.0;

    for left in 0..num_mutations {
        for right in 0..left {
            let similarity = sim_mat[left * num_mutations + right];
            similarity_total += similarity;

            if cluster_labels[left] == cluster_labels[right] {
                indicator_total += 1.0;
                indicator_similarity += similarity;
            }
        }
    }

    let combinations = (num_mutations * (num_mutations - 1) / 2) as f64;
    let z = (indicator_total * similarity_total) / combinations;
    let numerator = indicator_similarity - z;
    let denominator = 0.5 * (indicator_total + similarity_total) - z;

    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn merge_cluster_labels(
    cluster_labels: &[usize],
    left_cluster: usize,
    right_cluster: usize,
) -> Vec<usize> {
    let mut merged = cluster_labels.to_vec();

    for label in &mut merged {
        if *label == right_cluster {
            *label = left_cluster;
        }
    }

    let mut remap = HashMap::new();
    let mut next_label = 0usize;
    for label in &mut merged {
        let mapped = *remap.entry(*label).or_insert_with(|| {
            let current = next_label;
            next_label += 1;
            current
        });
        *label = mapped;
    }

    merged
}

fn refine_labels_by_merging(
    cluster_labels: &[usize],
    sim_mat: &[f64],
    num_mutations: usize,
) -> Vec<usize> {
    let mut best_labels = cluster_labels.to_vec();
    let mut best_score = compute_mpear(&best_labels, sim_mat, num_mutations);

    loop {
        let clusters = clusters_from_labels(&best_labels);
        if clusters.len() <= 1 {
            break;
        }

        let mut improved_labels = best_labels.clone();
        let mut improved_score = best_score;

        for left_cluster in 0..clusters.len() {
            for right_cluster in (left_cluster + 1)..clusters.len() {
                let avg_similarity = cluster_average_similarity(
                    &clusters[left_cluster],
                    &clusters[right_cluster],
                    sim_mat,
                    num_mutations,
                );
                if avg_similarity < MERGE_REFINEMENT_MIN_SIMILARITY {
                    continue;
                }

                let candidate = merge_cluster_labels(&best_labels, left_cluster, right_cluster);
                let candidate_score = compute_mpear(&candidate, sim_mat, num_mutations);
                if candidate_score > improved_score {
                    improved_score = candidate_score;
                    improved_labels = candidate;
                }
            }
        }

        if improved_score > best_score {
            best_score = improved_score;
            best_labels = improved_labels;
        } else {
            break;
        }
    }

    best_labels
}

pub fn cluster_with_mpear(
    sim_mat: &[f64],
    num_mutations: usize,
    max_clusters: usize,
) -> Vec<usize> {
    let max_clusters = max_clusters.min(num_mutations).max(1);
    let dist_mat = sim_mat.iter().map(|&value| 1.0 - value).collect::<Vec<_>>();
    let candidate_labels = average_linkage_labels(&dist_mat, num_mutations, max_clusters);

    let mut best_labels = vec![0usize; num_mutations];
    let mut best_mpear = 0.0;

    for num_clusters in 2..=max_clusters {
        if candidate_labels[num_clusters].is_empty() {
            continue;
        }

        let mpear = compute_mpear(&candidate_labels[num_clusters], sim_mat, num_mutations);
        if mpear > best_mpear {
            best_mpear = mpear;
            best_labels = candidate_labels[num_clusters].clone();
        }
    }

    refine_labels_by_merging(&best_labels, sim_mat, num_mutations)
}

pub fn build_result_from_trace(
    trace: &McmcTrace,
    data: &[SampleDataPoint],
    density: Density,
    num_mutations: usize,
    num_samples: usize,
    max_clusters: usize,
) -> Result<PcvResult, String> {
    if trace.num_samples == 0 {
        return Err("MCMC trace contains no saved samples".to_string());
    }

    let psm = posterior_similarity_matrix(trace, num_mutations);
    let labels = cluster_with_mpear(&psm, num_mutations, max_clusters);

    let mut cluster_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (mutation_index, &cluster_index) in labels.iter().enumerate() {
        cluster_members
            .entry(cluster_index)
            .or_default()
            .push(mutation_index);
    }

    let mut cluster_ids = cluster_members.keys().copied().collect::<Vec<_>>();
    cluster_ids.sort_unstable();

    let mut remap = HashMap::new();
    for (new_cluster, old_cluster) in cluster_ids.iter().enumerate() {
        remap.insert(*old_cluster, new_cluster);
    }

    let mut mutation_cluster_ids = vec![0_i32; num_mutations];
    let mut mutation_cluster_probs = vec![0.0; num_mutations];
    let mut mutation_sample_prevalence = vec![0.0; num_mutations * num_samples];
    let mut mutation_sample_prevalence_std = vec![0.0; num_mutations * num_samples];
    for mutation_index in 0..num_mutations {
        let old_cluster = labels[mutation_index];
        let new_cluster = *remap
            .get(&old_cluster)
            .ok_or_else(|| "cluster remapping failed".to_string())?;
        mutation_cluster_ids[mutation_index] = new_cluster as i32;

        let members = cluster_members
            .get(&old_cluster)
            .ok_or_else(|| "cluster member lookup failed".to_string())?;
        let avg_prob = members
            .iter()
            .map(|&other| psm[mutation_index * num_mutations + other])
            .sum::<f64>()
            / members.len().max(1) as f64;
        mutation_cluster_probs[mutation_index] = avg_prob;

        for sample_index in 0..num_samples {
            let offset = mutation_index * num_samples + sample_index;
            let mean = trace.ccf_sum[offset] / trace.num_samples as f64;
            let second = trace.ccf_sum_sq[offset] / trace.num_samples as f64;
            let variance = (second - mean * mean).max(0.0);
            mutation_sample_prevalence[offset] = mean;
            mutation_sample_prevalence_std[offset] = variance.sqrt();
        }
    }

    let used_k = cluster_ids.len();
    let mut cluster_sample_prevalence = vec![0.0; used_k * num_samples];
    let mut cluster_sample_prevalence_std = vec![0.0; used_k * num_samples];
    let mean_precision = (trace.precision_sum / trace.num_samples as f64).max(1e-6);

    for old_cluster in cluster_ids {
        let new_cluster = remap[&old_cluster];
        let members = &cluster_members[&old_cluster];
        for sample_index in 0..num_samples {
            let (mean, std) = compute_cluster_sample_posterior_stats(
                members,
                sample_index,
                data,
                num_samples,
                density,
                mean_precision,
            );
            let out_offset = new_cluster * num_samples + sample_index;
            cluster_sample_prevalence[out_offset] = mean;
            cluster_sample_prevalence_std[out_offset] = std;
        }
    }

    Ok(PcvResult {
        num_mutations,
        num_samples,
        num_clusters: used_k,
        num_saved_trace_samples: trace.num_samples,
        mutation_cluster_ids,
        mutation_cluster_probs,
        mutation_sample_prevalence,
        mutation_sample_prevalence_std,
        saved_mutation_sample_prevalence: trace.saved_ccf_trace.clone(),
        saved_precision_trace: trace.saved_precision_trace.clone(),
        cluster_sample_prevalence,
        cluster_sample_prevalence_std,
    })
}

fn compute_cluster_sample_posterior_stats(
    members: &[usize],
    sample_index: usize,
    data: &[SampleDataPoint],
    num_samples: usize,
    density: Density,
    precision: f64,
) -> (f64, f64) {
    let mut grid = Vec::with_capacity(POSTPROCESS_MESH_SIZE);
    let mut log_post = Vec::with_capacity(POSTPROCESS_MESH_SIZE);

    for mesh_index in 0..POSTPROCESS_MESH_SIZE {
        let cellular_prevalence = mesh_index as f64 / (POSTPROCESS_MESH_SIZE - 1) as f64;
        grid.push(cellular_prevalence);

        let mut total = 0.0;
        for &mutation_index in members {
            let datum = &data[mutation_index * num_samples + sample_index];
            total += match density {
                Density::Binomial => log_pyclone_binomial_pdf(datum, cellular_prevalence),
                Density::BetaBinomial => {
                    log_pyclone_beta_binomial_pdf(datum, cellular_prevalence, precision)
                }
            };
        }
        log_post.push(total);
    }

    let norm = log_sum_exp(&log_post);
    let probs = log_post
        .iter()
        .map(|value| (value - norm).exp())
        .collect::<Vec<_>>();
    let mean = grid
        .iter()
        .zip(probs.iter())
        .map(|(x, p)| x * p)
        .sum::<f64>();
    let second = grid
        .iter()
        .zip(probs.iter())
        .map(|(x, p)| x * x * p)
        .sum::<f64>();
    let variance = (second - mean * mean).max(0.0);

    (mean, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::{cluster_with_mpear, compute_mpear, refine_labels_by_merging};

    #[test]
    fn merge_refinement_collapses_unnecessary_split_when_mpear_improves() {
        let sim_mat = vec![
            1.0, 0.86, 0.84, 0.10, 0.86, 1.0, 0.82, 0.10, 0.84, 0.82, 1.0, 0.10, 0.10, 0.10, 0.10,
            1.0,
        ];

        let split_labels = vec![0, 1, 1, 2];
        let refined = refine_labels_by_merging(&split_labels, &sim_mat, 4);

        assert!(compute_mpear(&refined, &sim_mat, 4) > compute_mpear(&split_labels, &sim_mat, 4));
        assert_eq!(refined[0], refined[1]);
        assert_eq!(refined[1], refined[2]);
        assert_ne!(refined[0], refined[3]);
    }

    #[test]
    fn cluster_with_mpear_keeps_two_block_structure() {
        let sim_mat = vec![
            1.0, 0.95, 0.10, 0.10, 0.95, 1.0, 0.10, 0.10, 0.10, 0.10, 1.0, 0.90, 0.10, 0.10, 0.90,
            1.0,
        ];

        let labels = cluster_with_mpear(&sim_mat, 4, 4);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[2], labels[3]);
        assert_ne!(labels[0], labels[2]);
    }
}
