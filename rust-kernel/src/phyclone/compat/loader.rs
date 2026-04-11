use std::collections::{BTreeMap, BTreeSet};
use std::f64;

#[cfg(test)]
use rand::rngs::StdRng;
use rand::seq::index::sample;
#[cfg(test)]
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use crate::likelihood::compute_likelihood_grid_into;
use crate::preprocess::build_sample_data_point;
use crate::types::Density;

use super::data::{CompatDataPoint, CompatDataPointName};
use super::likelihood_grid::{build_ccf_grid, CompatGridConfig};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatInputRow {
    pub mutation_id: String,
    pub sample_id: String,
    pub ref_counts: i32,
    pub alt_counts: i32,
    pub major_cn: i32,
    pub minor_cn: i32,
    pub normal_cn: i32,
    pub tumour_content: f64,
    pub error_rate: f64,
    #[serde(default)]
    pub chrom: Option<String>,
    #[serde(default)]
    pub loss_prob: Option<f64>,
    #[serde(default)]
    pub cluster_id: Option<String>,
    #[serde(default)]
    pub outlier_prob: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CompatClusterRow {
    pub mutation_id: String,
    pub cluster_id: String,
    #[serde(default)]
    pub sample_id: Option<String>,
    #[serde(default)]
    pub chrom: Option<String>,
    #[serde(default)]
    pub cellular_prevalence: Option<f64>,
    #[serde(default)]
    pub outlier_prob: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompatLoaderConfig {
    pub density: Density,
    pub precision: f64,
    pub grid_size: usize,
    /// Global fallback outlier probability applied to any mutation that does not have a
    /// per-mutation or per-cluster `outlier_prob` override.  Mirrors PhyClone's default
    /// `--outlier-prob 0.0`.
    pub global_outlier_prob: f64,
    pub assign_loss_prob: bool,
    pub user_provided_loss_prob: bool,
    pub loss_prob: f64,
    pub high_loss_prob: f64,
}

impl Default for CompatLoaderConfig {
    fn default() -> Self {
        Self {
            density: Density::BetaBinomial,
            precision: 400.0,
            grid_size: 101,
            global_outlier_prob: 0.0,
            assign_loss_prob: false,
            user_provided_loss_prob: false,
            loss_prob: 0.0,
            high_loss_prob: 0.4,
        }
    }
}

#[derive(Clone)]
struct MutationPrepared {
    cluster_key: String,
    mutation_id: String,
    sample_likelihoods: Vec<Vec<f64>>, // sample x grid
    mutation_outlier_prob: f64,
}

#[cfg(test)]
pub fn build_compat_data_points(
    rows: &[CompatInputRow],
    cluster_rows: Option<&[CompatClusterRow]>,
    config: &CompatLoaderConfig,
) -> Result<Vec<CompatDataPoint>, String> {
    let mut rng = StdRng::seed_from_u64(rand::random::<u64>());
    build_compat_data_points_with_rng(rows, cluster_rows, config, &mut rng)
}

pub fn build_compat_data_points_with_rng<R: rand::Rng>(
    rows: &[CompatInputRow],
    cluster_rows: Option<&[CompatClusterRow]>,
    config: &CompatLoaderConfig,
    rng: &mut R,
) -> Result<Vec<CompatDataPoint>, String> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }

    validate_input_rows(rows)?;

    if config.precision <= 0.0 {
        return Err("precision must be > 0".to_string());
    }

    let ccf_grid = build_ccf_grid(CompatGridConfig {
        grid_size: config.grid_size,
    });
    let num_grid = ccf_grid.len();

    let sample_ids = collect_sorted_sample_ids(rows);
    if sample_ids.is_empty() {
        return Ok(Vec::new());
    }

    let cluster_map = build_cluster_map(cluster_rows)?;
    let cluster_outlier_probs =
        build_cluster_outlier_probs(rows, cluster_rows, &cluster_map, config, rng)?;

    let mut by_mutation: BTreeMap<String, Vec<&CompatInputRow>> = BTreeMap::new();
    for row in rows {
        by_mutation
            .entry(row.mutation_id.clone())
            .or_default()
            .push(row);
    }

    let mut prepared = Vec::new();
    for (mutation_id, mutation_rows) in by_mutation {
        let prepared_mutation = prepare_mutation(
            &mutation_id,
            &mutation_rows,
            &sample_ids,
            &cluster_map,
            &ccf_grid,
            config,
            &cluster_outlier_probs,
        )?;
        if let Some(value) = prepared_mutation {
            prepared.push(value);
        }
    }

    let mut grouped: BTreeMap<String, Vec<MutationPrepared>> = BTreeMap::new();
    for item in prepared {
        grouped
            .entry(item.cluster_key.clone())
            .or_default()
            .push(item);
    }

    let mut data_points = Vec::with_capacity(grouped.len());
    for (idx, (cluster_key, mutations)) in grouped.into_iter().enumerate() {
        let mut value = vec![vec![0.0_f64; num_grid]; sample_ids.len()];

        for mutation in &mutations {
            for (sample_index, sample_ll) in mutation.sample_likelihoods.iter().enumerate() {
                for (grid_index, ll) in sample_ll.iter().enumerate() {
                    value[sample_index][grid_index] += ll;
                }
            }
        }

        let size = mutations.len();
        // PhyClone compatibility: clustered DataPoint uses the first mutation's
        // outlier probability rather than averaging within the cluster.
        let base_outlier = mutations
            .first()
            .map(|m| m.mutation_outlier_prob)
            .unwrap_or(0.0);
        let (outlier_prob, outlier_prob_not) =
            compute_outlier_log_probs(base_outlier.clamp(0.0, 1.0), size);

        let mut dp_mutation_ids: Vec<String> =
            mutations.iter().map(|m| m.mutation_id.clone()).collect();
        dp_mutation_ids.sort();

        let outlier_marginal_prob = compute_outlier_marginal_prob(&value);

        data_points.push(CompatDataPoint {
            idx,
            name: CompatDataPointName::Str(cluster_key),
            mutation_ids: dp_mutation_ids,
            sample_ids: sample_ids.clone(),
            value,
            raw_outlier_prob: base_outlier.clamp(0.0, 1.0),
            outlier_prob,
            outlier_prob_not,
            outlier_marginal_prob,
            size,
        });
    }

    Ok(data_points)
}

fn collect_sorted_sample_ids(rows: &[CompatInputRow]) -> Vec<String> {
    let set: BTreeSet<String> = rows.iter().map(|row| row.sample_id.clone()).collect();
    set.into_iter().collect()
}

fn validate_input_rows(rows: &[CompatInputRow]) -> Result<(), String> {
    for (idx, row) in rows.iter().enumerate() {
        if row.mutation_id.trim().is_empty() {
            return Err(format!("row {}: mutation_id must be non-empty", idx));
        }
        if row.sample_id.trim().is_empty() {
            return Err(format!("row {}: sample_id must be non-empty", idx));
        }
        if row.ref_counts < 0 {
            return Err(format!("row {}: ref_counts must be >= 0", idx));
        }
        if row.alt_counts < 0 {
            return Err(format!("row {}: alt_counts must be >= 0", idx));
        }
        if row.major_cn < 0 {
            return Err(format!("row {}: major_cn must be >= 0", idx));
        }
        if row.minor_cn < 0 {
            return Err(format!("row {}: minor_cn must be >= 0", idx));
        }
        if row.normal_cn < 1 {
            return Err(format!("row {}: normal_cn must be >= 1", idx));
        }
        if !row.tumour_content.is_finite() || row.tumour_content < 0.0 {
            return Err(format!(
                "row {}: tumour_content must be finite and >= 0",
                idx
            ));
        }
        if !row.error_rate.is_finite() || row.error_rate < 0.0 {
            return Err(format!("row {}: error_rate must be finite and >= 0", idx));
        }
        if let Some(cluster_id) = &row.cluster_id {
            if cluster_id.trim().is_empty() {
                return Err(format!("row {}: cluster_id must be non-empty", idx));
            }
        }
        if let Some(outlier_prob) = row.outlier_prob {
            if !outlier_prob.is_finite() || outlier_prob < 0.0 {
                return Err(format!("row {}: outlier_prob must be finite and >= 0", idx));
            }
        }
        if let Some(loss_prob) = row.loss_prob {
            if !loss_prob.is_finite() || !(0.0..=1.0).contains(&loss_prob) {
                return Err(format!(
                    "row {}: loss_prob must be finite and in [0, 1]",
                    idx
                ));
            }
        }
    }

    Ok(())
}

fn validate_cluster_rows(rows: &[CompatClusterRow]) -> Result<(), String> {
    for (idx, row) in rows.iter().enumerate() {
        if row.mutation_id.trim().is_empty() {
            return Err(format!(
                "cluster row {}: mutation_id must be non-empty",
                idx
            ));
        }
        if row.cluster_id.trim().is_empty() {
            return Err(format!("cluster row {}: cluster_id must be non-empty", idx));
        }
        if let Some(outlier_prob) = row.outlier_prob {
            if !outlier_prob.is_finite() || outlier_prob < 0.0 {
                return Err(format!(
                    "cluster row {}: outlier_prob must be finite and >= 0",
                    idx
                ));
            }
        }
        if let Some(cellular_prevalence) = row.cellular_prevalence {
            if !cellular_prevalence.is_finite() || !(0.0..=1.0).contains(&cellular_prevalence) {
                return Err(format!(
                    "cluster row {}: cellular_prevalence must be finite and in [0, 1]",
                    idx
                ));
            }
        }
    }

    Ok(())
}

/// Groups mutation IDs and sample IDs by their resolved cluster key.
///
/// Does **not** apply the filtering used in [`build_compat_data_points`] (e.g. missing-sample
/// drop, duplicate-sample drop, `major_cn == 0` drop). This is intentional: the function is
/// used to build cluster summaries for the smoke-trace generator, where backward-compatible
/// behaviour (all mutations visible) is more important than PhyClone parity filtering.
///
/// **DEPRECATED**: This function is not currently used. Production path now uses
/// `build_trace_cluster_summaries_from_data_points` which applies full PhyClone-compatible
/// filtering. Kept for reference only; consider removing in a future cleanup pass.
fn build_cluster_map(
    cluster_rows: Option<&[CompatClusterRow]>,
) -> Result<BTreeMap<String, CompatClusterRow>, String> {
    let mut map: BTreeMap<String, CompatClusterRow> = BTreeMap::new();

    if let Some(rows) = cluster_rows {
        validate_cluster_rows(rows)?;

        for row in rows {
            if let Some(existing) = map.get(&row.mutation_id) {
                if existing.cluster_id != row.cluster_id {
                    return Err(format!(
                        "cluster_id mismatch for mutation {}",
                        row.mutation_id
                    ));
                }
            }
            map.insert(row.mutation_id.clone(), row.clone());
        }
    }

    Ok(map)
}

/// Represents a mutation with its cluster context and chromosome information.
/// Used for assign_loss_prob analysis.
#[derive(Clone, Debug)]
struct ClusterAssignmentRow {
    #[allow(dead_code)]
    mutation_id: String,
    cluster_id: String,
    sample_id: String,
    chrom: String,
    cellular_prevalence: Option<f64>,
}

/// Summary of cluster characteristics for lost-cluster detection.
#[derive(Clone, Debug)]
struct ClusterInfo {
    #[allow(dead_code)]
    cluster_id: String,
    num_mutations: usize,
    num_unique_chromosomes: usize,
    cell_prev_mean: f64,
}

/// Build assignment rows combining mutation data with cluster metadata.
fn build_cluster_assignment_rows(
    rows: &[CompatInputRow],
    cluster_rows: Option<&[CompatClusterRow]>,
    cluster_map: &BTreeMap<String, CompatClusterRow>,
) -> Vec<ClusterAssignmentRow> {
    let mut by_mutation_sample: BTreeMap<(String, String), &CompatClusterRow> = BTreeMap::new();
    let mut by_mutation: BTreeMap<String, &CompatClusterRow> = BTreeMap::new();

    if let Some(cluster_rows) = cluster_rows {
        for cluster_row in cluster_rows {
            if let Some(sample_id) = &cluster_row.sample_id {
                by_mutation_sample.insert(
                    (cluster_row.mutation_id.clone(), sample_id.clone()),
                    cluster_row,
                );
            }
            by_mutation
                .entry(cluster_row.mutation_id.clone())
                .or_insert(cluster_row);
        }
    }

    rows.iter()
        .map(|row| {
            let matched_cluster_row = by_mutation_sample
                .get(&(row.mutation_id.clone(), row.sample_id.clone()))
                .copied()
                .or_else(|| by_mutation.get(&row.mutation_id).copied())
                .or_else(|| cluster_map.get(&row.mutation_id));

            let cluster_id = matched_cluster_row
                .map(|c| c.cluster_id.clone())
                .or_else(|| row.cluster_id.clone())
                .unwrap_or_else(|| row.mutation_id.clone());

            let cellular_prevalence = matched_cluster_row.and_then(|c| c.cellular_prevalence);

            let chrom = matched_cluster_row
                .and_then(|c| c.chrom.as_deref())
                .or(row.chrom.as_deref())
                .unwrap_or("unknown")
                .to_string();

            ClusterAssignmentRow {
                mutation_id: row.mutation_id.clone(),
                cluster_id,
                sample_id: row.sample_id.clone(),
                chrom,
                cellular_prevalence,
            }
        })
        .collect()
}

/// Determine the truncal (clonal) cluster.
/// PhyClone semantics: For each sample, find cluster(s) with max cellular_prevalence.
/// If one cluster leads in all samples, use it. Otherwise, use mean cellular_prevalence.
fn define_truncal_cluster(rows: &[ClusterAssignmentRow]) -> Option<String> {
    if rows.is_empty() {
        return None;
    }

    // Group by (sample_id, cluster_id) to compute max prevalence per sample.
    let mut sample_cluster_prevs: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    for row in rows {
        if let Some(prev) = row.cellular_prevalence {
            sample_cluster_prevs
                .entry((row.sample_id.clone(), row.cluster_id.clone()))
                .or_default()
                .push(prev);
        }
    }

    // For each sample, find max prevalence.
    let mut sample_max_prevs: BTreeMap<String, f64> = BTreeMap::new();
    for ((sample_id, _), prevs) in &sample_cluster_prevs {
        let max = prevs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        sample_max_prevs
            .entry(sample_id.clone())
            .and_modify(|v| *v = v.max(max))
            .or_insert(max);
    }

    // Collect clusters that achieve max in each sample.
    let mut cluster_scores: BTreeMap<String, usize> = BTreeMap::new();
    for ((sample_id, cluster_id), prevs) in &sample_cluster_prevs {
        if let Some(&max_in_sample) = sample_max_prevs.get(sample_id) {
            let actual_max = prevs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if (actual_max - max_in_sample).abs() < 1e-12 {
                *cluster_scores.entry(cluster_id.clone()).or_default() += 1;
            }
        }
    }

    // If one cluster appears in all samples, it's truncal.
    let num_samples = sample_max_prevs.len();
    for (cluster_id, count) in &cluster_scores {
        if *count == num_samples {
            return Some(cluster_id.clone());
        }
    }

    // Fallback: pick cluster with highest mean cellular_prevalence.
    let mut cluster_means: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    for row in rows {
        if let Some(prev) = row.cellular_prevalence {
            let entry = cluster_means
                .entry(row.cluster_id.clone())
                .or_insert((0.0, 0));
            entry.0 += prev;
            entry.1 += 1;
        }
    }

    cluster_means
        .into_iter()
        .max_by(|(_, (sum_a, count_a)), (_, (sum_b, count_b))| {
            let mean_a = sum_a / (*count_a as f64);
            let mean_b = sum_b / (*count_b as f64);
            mean_a
                .partial_cmp(&mean_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(cluster_id, _)| cluster_id)
}

/// Get chromosome array for a cluster with one entry per unique mutation.
fn get_truncal_chrom_array(rows: &[ClusterAssignmentRow], cluster_id: &str) -> Vec<String> {
    let mut mutation_to_chrom: BTreeMap<String, String> = BTreeMap::new();
    for row in rows.iter().filter(|row| row.cluster_id == cluster_id) {
        mutation_to_chrom
            .entry(row.mutation_id.clone())
            .or_insert_with(|| row.chrom.clone());
    }
    mutation_to_chrom.into_values().collect()
}

/// PhyClone's permutation-based lost-cluster detection.
/// Returns Vec of cluster IDs deemed "possibly lost".
fn define_possibly_lost_clusters(
    all_rows: &[ClusterAssignmentRow],
    truncal_cluster_id: &str,
    cluster_infos: &BTreeMap<String, ClusterInfo>,
    rng: &mut (impl rand::Rng + ?Sized),
) -> Vec<String> {
    const TEST_ITERS: usize = 10_000;
    const P_VALUE_THRESHOLD: f64 = 0.01;
    // Match PhyClone default min_clust_size.
    const MIN_CLUSTER_SIZE: usize = 4;

    let truncal_chrom_arr = get_truncal_chrom_array(all_rows, truncal_cluster_id);
    if truncal_chrom_arr.is_empty() {
        return Vec::new();
    }

    let mut lost = Vec::new();

    for (cluster_id, info) in cluster_infos {
        if cluster_id == truncal_cluster_id {
            continue;
        }
        if info.num_mutations < MIN_CLUSTER_SIZE {
            continue;
        }

        let cluster_chrom_arr = get_truncal_chrom_array(all_rows, cluster_id);
        let observed_unique = cluster_chrom_arr.iter().collect::<BTreeSet<_>>().len();

        // Resample truncal_chrom_arr to cluster size and count unique chromosomes.
        let mut num_unique_sum = 0;
        let mut samples_fewer = 0;

        for _ in 0..TEST_ITERS {
            let sampled =
                resample_without_replacement(&truncal_chrom_arr, cluster_chrom_arr.len(), rng);
            let num_unique = sampled.iter().collect::<BTreeSet<_>>().len();
            num_unique_sum += num_unique;
            if num_unique < observed_unique {
                samples_fewer += 1;
            }
        }

        let p_value = if samples_fewer == 0 {
            1.0 / (TEST_ITERS as f64)
        } else {
            (samples_fewer as f64) / (TEST_ITERS as f64)
        };

        let estimate = (num_unique_sum as f64 / TEST_ITERS as f64) / (observed_unique as f64);

        // Lost if p < 0.01 and observed unique > expected.
        if p_value < P_VALUE_THRESHOLD && estimate > 1.0 {
            lost.push(cluster_id.clone());
        }
    }

    lost
}

/// Resample without replacement.
fn resample_without_replacement<T: Clone, R: rand::Rng + ?Sized>(
    arr: &[T],
    n: usize,
    rng: &mut R,
) -> Vec<T> {
    if arr.is_empty() || n == 0 {
        return Vec::new();
    }

    // Match PhyClone's numpy path:
    // if target size is larger than source, first np.resize to length n,
    // then sample without replacement from that resized population.
    let population: Vec<T> = if n > arr.len() {
        let mut resized = Vec::with_capacity(n);
        while resized.len() < n {
            let remaining = n - resized.len();
            let take = remaining.min(arr.len());
            resized.extend(arr[..take].iter().cloned());
        }
        resized
    } else {
        arr.to_vec()
    };

    let sample_size = n.min(population.len());
    let sampled_indices = sample(rng, population.len(), sample_size);
    sampled_indices
        .into_iter()
        .map(|idx| population[idx].clone())
        .collect()
}

/// Compute cluster-level outlier probabilities using assign_loss_prob semantics.
/// Returns a map of cluster_id -> outlier_prob.
fn build_cluster_outlier_probs(
    rows: &[CompatInputRow],
    cluster_rows: Option<&[CompatClusterRow]>,
    cluster_map: &BTreeMap<String, CompatClusterRow>,
    config: &CompatLoaderConfig,
    rng: &mut (impl rand::Rng + ?Sized),
) -> Result<BTreeMap<String, f64>, String> {
    if !config.assign_loss_prob {
        return Ok(BTreeMap::new());
    }

    let assignment_rows = build_cluster_assignment_rows(rows, cluster_rows, cluster_map);

    if config.assign_loss_prob {
        if assignment_rows
            .iter()
            .any(|row| row.cellular_prevalence.is_none())
        {
            return Err(
                "assign_loss_prob requires cellular_prevalence for each mutation/sample (cluster-file)"
                    .to_string(),
            );
        }

        if assignment_rows.iter().any(|row| row.chrom == "unknown") {
            return Err(
                "assign_loss_prob requires chromosome information (chrom column in input or cluster-file)"
                    .to_string(),
            );
        }
    }

    // Compute cluster info (num_mutations, num_unique_chromosomes, mean cellular_prevalence).
    let mut cluster_infos: BTreeMap<String, ClusterInfo> = BTreeMap::new();
    let mut cluster_mutations: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut cluster_chroms: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut cluster_prevs: BTreeMap<String, (f64, usize)> = BTreeMap::new();

    for row in &assignment_rows {
        cluster_infos
            .entry(row.cluster_id.clone())
            .or_insert(ClusterInfo {
                cluster_id: row.cluster_id.clone(),
                num_mutations: 0,
                num_unique_chromosomes: 0,
                cell_prev_mean: 0.0,
            });

        cluster_mutations
            .entry(row.cluster_id.clone())
            .or_default()
            .insert(row.mutation_id.clone());

        cluster_chroms
            .entry(row.cluster_id.clone())
            .or_default()
            .insert(row.chrom.clone());

        if let Some(prev) = row.cellular_prevalence {
            let (sum, count) = cluster_prevs
                .entry(row.cluster_id.clone())
                .or_insert((0.0, 0));
            *sum += prev;
            *count += 1;
        }
    }

    // Update ClusterInfo with computed values.
    for (cluster_id, info) in &mut cluster_infos {
        info.num_mutations = cluster_mutations
            .get(cluster_id)
            .map(|s| s.len())
            .unwrap_or(0);
        info.num_unique_chromosomes = cluster_chroms.get(cluster_id).map(|s| s.len()).unwrap_or(0);
        if let Some((sum, count)) = cluster_prevs.get(cluster_id) {
            info.cell_prev_mean = sum / (*count as f64);
        }
    }

    // Determine truncal cluster.
    let truncal_cluster_id = match define_truncal_cluster(&assignment_rows) {
        Some(id) => id,
        None => {
            // No clear truncal cluster; all clusters get global_outlier_prob.
            let mut result = BTreeMap::new();
            for cluster_id in cluster_infos.keys() {
                result.insert(cluster_id.clone(), config.global_outlier_prob);
            }
            return Ok(result);
        }
    };

    // Identify lost clusters via permutation test.
    let lost_cluster_ids =
        define_possibly_lost_clusters(&assignment_rows, &truncal_cluster_id, &cluster_infos, rng);

    // Assign outlier probabilities.
    let mut result = BTreeMap::new();
    for cluster_id in cluster_infos.keys() {
        let outlier_prob = if lost_cluster_ids.contains(cluster_id) {
            config.high_loss_prob
        } else {
            config.global_outlier_prob
        };
        result.insert(cluster_id.clone(), outlier_prob);
    }

    Ok(result)
}

fn prepare_mutation(
    mutation_id: &str,
    mutation_rows: &[&CompatInputRow],
    sample_ids: &[String],
    cluster_map: &BTreeMap<String, CompatClusterRow>,
    ccf_grid: &[f64],
    config: &CompatLoaderConfig,
    cluster_outlier_probs: &BTreeMap<String, f64>,
) -> Result<Option<MutationPrepared>, String> {
    let mut by_sample: BTreeMap<&str, &CompatInputRow> = BTreeMap::new();
    let mut has_duplicate_sample = false;
    for row in mutation_rows {
        if by_sample.insert(row.sample_id.as_str(), row).is_some() {
            has_duplicate_sample = true;
            break;
        }
    }

    if has_duplicate_sample {
        // Match PhyClone loader behavior: drop mutations with duplicate sample rows.
        return Ok(None);
    }

    if sample_ids
        .iter()
        .any(|sample_id| !by_sample.contains_key(sample_id.as_str()))
    {
        // Keep parity direction with PhyClone filtering: drop mutations not present in all samples.
        return Ok(None);
    }

    if mutation_rows.iter().any(|row| row.major_cn == 0) {
        return Ok(None);
    }

    let mut sample_likelihoods = Vec::with_capacity(sample_ids.len());
    for sample_id in sample_ids {
        let row = by_sample
            .get(sample_id.as_str())
            .ok_or_else(|| "internal sample lookup failure".to_string())?;

        let pcv_row = crate::abi::PcvRow {
            mutation_index: 0,
            sample_index: 0,
            ref_counts: row.ref_counts,
            alt_counts: row.alt_counts,
            major_cn: row.major_cn,
            minor_cn: row.minor_cn,
            normal_cn: row.normal_cn,
            tumour_content: row.tumour_content,
            error_rate: row.error_rate,
        };

        let sample_data = build_sample_data_point(&pcv_row)?;
        let mut likelihood = vec![0.0_f64; ccf_grid.len()];
        compute_likelihood_grid_into(
            &sample_data,
            ccf_grid,
            config.density,
            config.precision,
            &mut likelihood,
        )?;
        sample_likelihoods.push(likelihood);
    }

    // PhyClone semantics: grouping is driven exclusively by the cluster file.
    // Input-row cluster_id is intentionally ignored here; if no cluster file is provided,
    // each mutation becomes its own DataPoint (one mutation per DataPoint).
    let cluster_key = if let Some(cluster_row) = cluster_map.get(mutation_id) {
        cluster_row.cluster_id.clone()
    } else {
        mutation_id.to_string()
    };

    // user_provided_loss_prob: delegate entirely to cluster_row.outlier_prob (PhyClone-compatible).
    // Validation that a cluster file is present is enforced at the FFI layer before we get here.
    // Input-row loss_prob is intentionally ignored in this mode.
    //
    // assign_loss_prob: use pre-computed cluster-level outlier probabilities.
    // These are derived from permutation-based lost-cluster detection.
    let derived_loss_prob = if config.assign_loss_prob {
        // cluster_key is the resolved cluster identifier for this mutation.
        let cluster_key = if let Some(cluster_row) = cluster_map.get(mutation_id) {
            cluster_row.cluster_id.clone()
        } else {
            mutation_id.to_string()
        };
        cluster_outlier_probs.get(&cluster_key).copied()
    } else {
        None
    };

    let mutation_outlier_prob = cluster_map
        .get(mutation_id)
        .and_then(|row| row.outlier_prob)
        .or(derived_loss_prob)
        .unwrap_or(config.global_outlier_prob)
        .clamp(0.0, 1.0);

    Ok(Some(MutationPrepared {
        cluster_key,
        mutation_id: mutation_id.to_string(),
        sample_likelihoods,
        mutation_outlier_prob,
    }))
}

fn compute_outlier_log_probs(outlier_prob: f64, cluster_size: usize) -> (f64, f64) {
    let p = outlier_prob.clamp(0.0, 1.0);
    let eps = f64::MIN_POSITIVE;
    let k = cluster_size as f64;

    if p == 0.0 {
        (eps.ln() * k, 0.0)
    } else if p == 1.0 {
        (0.0, eps.ln() * k)
    } else {
        (p.ln() * k, (1.0 - p).ln() * k)
    }
}

/// Compute `outlier_marginal_prob` from a likelihood grid, mirroring PhyClone's
/// `DataPoint.__init__` calculation:
///
/// ```text
/// log_prior = -log(grid_size)
/// sub_comp[s][g] = value[s][g] + log_prior
/// sub_comp[s][g] = cumulative_logsumexp(sub_comp[s])[g]
/// sub_comp[s][g] += log_prior
/// result = sum_s logsumexp(sub_comp[s])
/// ```
///
/// `value` is a `samples x grid` matrix of log likelihoods.
fn compute_outlier_marginal_prob(value: &[Vec<f64>]) -> f64 {
    if value.is_empty() {
        return f64::NEG_INFINITY;
    }
    let num_grid = value[0].len();
    if num_grid == 0 {
        return f64::NEG_INFINITY;
    }

    let log_prior = -(num_grid as f64).ln();

    let mut total = 0.0_f64;
    for sample_row in value {
        // sub_comp[g] = sample_row[g] + log_prior, then cumulative logsumexp
        let mut sub_comp = vec![0.0_f64; num_grid];
        let mut running_max = f64::NEG_INFINITY;
        let mut running_sum_exp = 0.0_f64;
        for (g, &ll) in sample_row.iter().enumerate() {
            let v = ll + log_prior;
            if v > running_max {
                // Shift the running sum to the new max
                running_sum_exp *= (running_max - v).exp();
                running_max = v;
            }
            running_sum_exp += (v - running_max).exp();
            sub_comp[g] = running_max + running_sum_exp.ln() + log_prior;
        }

        // logsumexp over sub_comp[s]
        let max = sub_comp.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = if max.is_finite() {
            max + sub_comp.iter().map(|&x| (x - max).exp()).sum::<f64>().ln()
        } else {
            f64::NEG_INFINITY
        };
        total += lse;
    }

    total
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;

    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use serde::Deserialize;

    use super::{
        build_compat_data_points, resample_without_replacement, CompatClusterRow, CompatInputRow,
        CompatLoaderConfig,
    };
    use crate::phyclone::compat::likelihood_grid::{build_ccf_grid, CompatGridConfig};
    use crate::types::Density;

    const ORACLE_FIXTURE_BETA_CLUSTERED: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/phyclone/compat/testdata/phyclone_oracle_fixture.json"
    );
    const ORACLE_FIXTURE_BETA_MULTI_MUTATION_ONE_CLUSTER: &str = ORACLE_FIXTURE_BETA_CLUSTERED;
    const ORACLE_FIXTURE_BINOMIAL_CLUSTERED: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/phyclone/compat/testdata/phyclone_oracle_fixture_binomial.json"
    );
    const ORACLE_FIXTURE_BETA_OUTLIER_SPLIT: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/phyclone/compat/testdata/phyclone_oracle_fixture_outlier_split.json"
    );
    const ORACLE_FIXTURE_BETA_CN_PURITY: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/phyclone/compat/testdata/phyclone_oracle_fixture_cn_purity.json"
    );
    const ORACLE_FIXTURE_BETA_PRECISION_50: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/phyclone/compat/testdata/phyclone_oracle_fixture_precision50.json"
    );

    #[derive(Debug, Deserialize)]
    struct OracleFixture {
        config: OracleConfig,
        rows: Vec<OracleInputRow>,
        #[serde(default)]
        cluster_rows: Vec<OracleClusterRow>,
        oracle: OraclePayload,
    }

    #[derive(Debug, Deserialize)]
    struct OracleConfig {
        density: String,
        precision: f64,
        grid_size: usize,
    }

    #[derive(Debug, Deserialize)]
    struct OracleInputRow {
        mutation_id: String,
        sample_id: String,
        ref_counts: i32,
        alt_counts: i32,
        major_cn: i32,
        minor_cn: i32,
        normal_cn: i32,
        tumour_content: f64,
        error_rate: f64,
        #[serde(default)]
        chrom: Option<String>,
        #[serde(default)]
        loss_prob: Option<f64>,
        cluster_id: Option<String>,
        outlier_prob: Option<f64>,
    }

    #[derive(Debug, Deserialize)]
    struct OracleClusterRow {
        mutation_id: String,
        cluster_id: String,
        #[serde(default)]
        cellular_prevalence: Option<f64>,
        outlier_prob: Option<f64>,
    }

    #[derive(Debug, Deserialize)]
    struct OraclePayload {
        ccf_grid: Vec<f64>,
        datapoints: Vec<OracleDataPoint>,
    }

    #[derive(Debug, Deserialize)]
    struct OracleDataPoint {
        name: String,
        value: Vec<Vec<f64>>,
        #[serde(default)]
        outlier_prob_log: Option<f64>,
        #[serde(default)]
        outlier_prob_not_log: Option<f64>,
        size: usize,
    }

    fn parse_density(value: &str) -> Density {
        match value {
            "binomial" => Density::Binomial,
            "beta-binomial" => Density::BetaBinomial,
            other => panic!("unsupported density in fixture: {}", other),
        }
    }

    fn assert_close(lhs: f64, rhs: f64, tol: f64, msg: &str) {
        let delta = (lhs - rhs).abs();
        assert!(
            delta <= tol,
            "{}: left={} right={} delta={} tol={}",
            msg,
            lhs,
            rhs,
            delta,
            tol
        );
    }

    fn base_row(mutation_id: &str, sample_id: &str, alt_counts: i32) -> CompatInputRow {
        CompatInputRow {
            mutation_id: mutation_id.to_string(),
            sample_id: sample_id.to_string(),
            ref_counts: 30 - alt_counts,
            alt_counts,
            major_cn: 1,
            minor_cn: 1,
            normal_cn: 2,
            tumour_content: 1.0,
            error_rate: 1e-3,
            chrom: None,
            loss_prob: None,
            cluster_id: None,
            outlier_prob: None,
        }
    }

    #[test]
    fn drops_mutations_missing_any_sample() {
        let rows = vec![
            base_row("m0", "s0", 5),
            base_row("m0", "s1", 6),
            base_row("m1", "s0", 7), // m1 missing s1
        ];

        let points = build_compat_data_points(&rows, None, &CompatLoaderConfig::default()).unwrap();
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].size, 1);
    }

    #[test]
    fn errors_on_invalid_input_row_values() {
        let mut invalid = base_row("m0", "s0", 5);
        invalid.normal_cn = 0;

        let err = build_compat_data_points(&[invalid], None, &CompatLoaderConfig::default())
            .expect_err("invalid row must error");
        assert!(err.contains("normal_cn"));
    }

    #[test]
    fn errors_on_empty_required_string_fields() {
        let mut invalid = base_row("m0", "s0", 5);
        invalid.mutation_id = "   ".to_string();

        let err = build_compat_data_points(&[invalid], None, &CompatLoaderConfig::default())
            .expect_err("invalid row must error");
        assert!(err.contains("mutation_id"));
    }

    #[test]
    fn errors_on_invalid_cluster_rows() {
        let rows = vec![base_row("m0", "s0", 5), base_row("m0", "s1", 6)];
        let clusters = vec![CompatClusterRow {
            mutation_id: "m0".to_string(),
            cluster_id: "c0".to_string(),
            sample_id: None,
            chrom: None,
            cellular_prevalence: None,
            outlier_prob: Some(-0.1),
        }];

        let err = build_compat_data_points(&rows, Some(&clusters), &CompatLoaderConfig::default())
            .expect_err("invalid cluster row must error");
        assert!(err.contains("outlier_prob"));
    }

    #[test]
    fn errors_on_invalid_loss_prob_in_input_row() {
        let mut invalid = base_row("m0", "s0", 5);
        invalid.loss_prob = Some(1.2);

        let err = build_compat_data_points(&[invalid], None, &CompatLoaderConfig::default())
            .expect_err("invalid row must error");
        assert!(err.contains("loss_prob"));
    }

    #[test]
    fn errors_on_invalid_cellular_prevalence_in_cluster_row() {
        let rows = vec![base_row("m0", "s0", 5), base_row("m0", "s1", 6)];
        let clusters = vec![CompatClusterRow {
            mutation_id: "m0".to_string(),
            cluster_id: "c0".to_string(),
            sample_id: None,
            chrom: None,
            cellular_prevalence: Some(1.5),
            outlier_prob: None,
        }];

        let err = build_compat_data_points(&rows, Some(&clusters), &CompatLoaderConfig::default())
            .expect_err("invalid cluster row must error");
        assert!(err.contains("cellular_prevalence"));
    }

    #[test]
    fn drops_mutations_with_major_cn_zero() {
        let mut bad = base_row("m0", "s0", 5);
        bad.major_cn = 0;

        let rows = vec![bad, base_row("m0", "s1", 6)];
        let points = build_compat_data_points(&rows, None, &CompatLoaderConfig::default()).unwrap();
        assert!(points.is_empty());
    }

    #[test]
    fn drops_mutations_with_duplicate_mutation_sample_pair() {
        let rows = vec![
            base_row("m0", "s0", 5),
            base_row("m0", "s0", 6),
            base_row("m0", "s1", 7),
            base_row("m1", "s0", 4),
            base_row("m1", "s1", 5),
        ];

        let points = build_compat_data_points(&rows, None, &CompatLoaderConfig::default()).unwrap();
        assert_eq!(points.len(), 1);
        match &points[0].name {
            super::CompatDataPointName::Str(value) => assert_eq!(value, "m1"),
            _ => panic!("expected string datapoint name"),
        }
    }

    /// Without a cluster file, input-row cluster_id is ignored: each mutation becomes its own
    /// DataPoint (PhyClone semantics). This test verifies that two mutations with the same
    /// cluster_id in the input TSV are NOT merged when no cluster file is provided.
    #[test]
    fn input_row_cluster_id_ignored_without_cluster_file() {
        let mut a0 = base_row("m0", "s0", 5);
        let mut a1 = base_row("m0", "s1", 6);
        let mut b0 = base_row("m1", "s0", 4);
        let mut b1 = base_row("m1", "s1", 5);

        // Both mutations share the same cluster_id in the input rows, but no cluster file is given.
        a0.cluster_id = Some("c0".to_string());
        a1.cluster_id = Some("c0".to_string());
        b0.cluster_id = Some("c0".to_string());
        b1.cluster_id = Some("c0".to_string());

        let rows = vec![a0, a1, b0, b1];
        let points = build_compat_data_points(
            &rows,
            None,
            &CompatLoaderConfig {
                density: Density::Binomial,
                precision: 400.0,
                grid_size: 5,
                global_outlier_prob: 1e-4,
                ..CompatLoaderConfig::default()
            },
        )
        .unwrap();

        // Without a cluster file, each mutation is its own DataPoint.
        assert_eq!(
            points.len(),
            2,
            "expected 2 separate DataPoints (one per mutation)"
        );
        assert_eq!(points[0].size, 1);
        assert_eq!(points[1].size, 1);
    }

    /// With a cluster file, mutations sharing the same cluster_id ARE merged into one DataPoint.
    #[test]
    fn cluster_file_merges_mutations_into_one_datapoint() {
        let rows = vec![
            base_row("m0", "s0", 5),
            base_row("m0", "s1", 6),
            base_row("m1", "s0", 4),
            base_row("m1", "s1", 5),
        ];
        let clusters = vec![
            CompatClusterRow {
                mutation_id: "m0".to_string(),
                cluster_id: "c0".to_string(),
                sample_id: None,
                chrom: None,
                cellular_prevalence: None,
                outlier_prob: None,
            },
            CompatClusterRow {
                mutation_id: "m1".to_string(),
                cluster_id: "c0".to_string(),
                sample_id: None,
                chrom: None,
                cellular_prevalence: None,
                outlier_prob: None,
            },
        ];
        let points = build_compat_data_points(
            &rows,
            Some(&clusters),
            &CompatLoaderConfig {
                density: Density::Binomial,
                precision: 400.0,
                grid_size: 5,
                global_outlier_prob: 1e-4,
                ..CompatLoaderConfig::default()
            },
        )
        .unwrap();

        assert_eq!(points.len(), 1, "expected 1 merged DataPoint");
        assert_eq!(points[0].size, 2);
        assert_eq!(points[0].value.len(), 2); // samples
        assert_eq!(points[0].value[0].len(), 5); // grid
    }

    #[test]
    fn prefers_cluster_file_mapping_and_outlier_prob() {
        let rows = vec![
            base_row("m0", "s0", 5),
            base_row("m0", "s1", 6),
            base_row("m1", "s0", 7),
            base_row("m1", "s1", 8),
        ];
        let clusters = vec![
            CompatClusterRow {
                mutation_id: "m0".to_string(),
                cluster_id: "cluster-A".to_string(),
                sample_id: None,
                chrom: None,
                cellular_prevalence: None,
                outlier_prob: Some(0.2),
            },
            CompatClusterRow {
                mutation_id: "m1".to_string(),
                cluster_id: "cluster-A".to_string(),
                sample_id: None,
                chrom: None,
                cellular_prevalence: None,
                outlier_prob: Some(0.2),
            },
        ];

        let points =
            build_compat_data_points(&rows, Some(&clusters), &CompatLoaderConfig::default())
                .unwrap();

        assert_eq!(points.len(), 1);
        match &points[0].name {
            super::CompatDataPointName::Str(value) => assert_eq!(value, "cluster-A"),
            _ => panic!("expected string datapoint name"),
        }
        let expected_log = 0.2_f64.ln() * 2.0;
        assert!((points[0].outlier_prob - expected_log).abs() < 1e-12);
    }

    #[test]
    fn clustered_outlier_prob_uses_first_mutation_value_not_mean() {
        let rows = vec![
            base_row("m0", "s0", 5),
            base_row("m0", "s1", 6),
            base_row("m1", "s0", 7),
            base_row("m1", "s1", 8),
        ];
        let clusters = vec![
            CompatClusterRow {
                mutation_id: "m0".to_string(),
                cluster_id: "cluster-A".to_string(),
                sample_id: None,
                chrom: None,
                cellular_prevalence: None,
                outlier_prob: Some(0.2),
            },
            CompatClusterRow {
                mutation_id: "m1".to_string(),
                cluster_id: "cluster-A".to_string(),
                sample_id: None,
                chrom: None,
                cellular_prevalence: None,
                outlier_prob: Some(0.8),
            },
        ];

        let points =
            build_compat_data_points(&rows, Some(&clusters), &CompatLoaderConfig::default())
                .expect("loader should succeed");

        assert_eq!(points.len(), 1);

        // Clustered datapoint size is 2, and the first mutation in cluster order is m0.
        let expected_log = 0.2_f64.ln() * 2.0;
        assert!(
            (points[0].outlier_prob - expected_log).abs() < 1e-12,
            "expected first mutation outlier_prob=0.2 to drive clustered outlier_prob, got {}",
            points[0].outlier_prob
        );
    }

    #[test]
    fn ignores_input_row_outlier_prob_without_loss_flags() {
        let mut r0 = base_row("m0", "s0", 5);
        let mut r1 = base_row("m0", "s1", 6);
        r0.outlier_prob = Some(0.3);
        r1.outlier_prob = Some(0.3);

        let points = build_compat_data_points(
            &[r0, r1],
            None,
            &CompatLoaderConfig {
                global_outlier_prob: 0.07,
                ..CompatLoaderConfig::default()
            },
        )
        .expect("loader should succeed");
        assert_eq!(points.len(), 1);

        let expected_log = 0.07_f64.ln();
        assert!((points[0].outlier_prob - expected_log).abs() < 1e-12);
    }

    // --- user_provided_loss_prob tests ---
    // PhyClone-compatible semantics: the cluster file's outlier_prob column is used.
    // Input-row loss_prob is ignored; the "no cluster file" guard lives in the FFI layer.

    #[test]
    fn user_provided_loss_prob_uses_cluster_outlier_prob() {
        let rows = vec![base_row("m0", "s0", 5), base_row("m0", "s1", 6)];
        let clusters = vec![CompatClusterRow {
            mutation_id: "m0".to_string(),
            cluster_id: "cloneA".to_string(),
            sample_id: None,
            chrom: None,
            cellular_prevalence: None,
            outlier_prob: Some(0.05),
        }];

        let points = build_compat_data_points(
            &rows,
            Some(&clusters),
            &CompatLoaderConfig {
                user_provided_loss_prob: true,
                global_outlier_prob: 0.0,
                ..CompatLoaderConfig::default()
            },
        )
        .expect("loader should succeed");
        assert_eq!(points.len(), 1);

        let expected_log = 0.05_f64.ln();
        assert!(
            (points[0].outlier_prob - expected_log).abs() < 1e-12,
            "expected cluster outlier_prob=0.05, got {}",
            points[0].outlier_prob
        );
    }

    #[test]
    fn user_provided_loss_prob_falls_back_to_global_when_cluster_has_no_outlier_prob() {
        let rows = vec![base_row("m0", "s0", 5), base_row("m0", "s1", 6)];
        let clusters = vec![CompatClusterRow {
            mutation_id: "m0".to_string(),
            cluster_id: "cloneA".to_string(),
            sample_id: None,
            chrom: None,
            cellular_prevalence: None,
            outlier_prob: None, // no per-cluster override
        }];

        let points = build_compat_data_points(
            &rows,
            Some(&clusters),
            &CompatLoaderConfig {
                user_provided_loss_prob: true,
                global_outlier_prob: 0.07,
                ..CompatLoaderConfig::default()
            },
        )
        .expect("loader should succeed");
        assert_eq!(points.len(), 1);

        let expected_log = 0.07_f64.ln();
        assert!(
            (points[0].outlier_prob - expected_log).abs() < 1e-12,
            "expected global_outlier_prob=0.07, got {}",
            points[0].outlier_prob
        );
    }

    #[test]
    fn user_provided_loss_prob_ignores_input_row_loss_prob() {
        // Input-row loss_prob must NOT influence the outlier probability even if present.
        // The cluster file's outlier_prob (or global fallback) is the only source.
        let mut r0 = base_row("m0", "s0", 5);
        let mut r1 = base_row("m0", "s1", 6);
        r0.loss_prob = Some(0.99); // should be ignored
        r1.loss_prob = Some(0.99);

        let clusters = vec![CompatClusterRow {
            mutation_id: "m0".to_string(),
            cluster_id: "cloneA".to_string(),
            sample_id: None,
            chrom: None,
            cellular_prevalence: None,
            outlier_prob: Some(0.03),
        }];

        let points = build_compat_data_points(
            &[r0, r1],
            Some(&clusters),
            &CompatLoaderConfig {
                user_provided_loss_prob: true,
                global_outlier_prob: 0.01,
                ..CompatLoaderConfig::default()
            },
        )
        .expect("loader should succeed");
        assert_eq!(points.len(), 1);

        // Must use cluster outlier_prob=0.03, not input-row loss_prob=0.99.
        let expected_log = 0.03_f64.ln();
        assert!(
            (points[0].outlier_prob - expected_log).abs() < 1e-12,
            "input-row loss_prob must not override cluster outlier_prob, got {}",
            points[0].outlier_prob
        );
    }

    // --- assign_loss_prob tests (Phase L2) ---
    // PhyClone-compatible semantics: uses permutation-based lost-cluster detection.

    #[test]
    fn assign_loss_prob_requires_cellular_prevalence_per_mutation_sample() {
        // assign_loss_prob now requires cellular_prevalence metadata.
        let mut r0 = base_row("m0", "s0", 5);
        let mut r1 = base_row("m0", "s1", 6);
        r0.chrom = Some("chrX".to_string());
        r1.chrom = Some("chrX".to_string());

        let clusters = vec![CompatClusterRow {
            mutation_id: "m0".to_string(),
            cluster_id: "cloneA".to_string(),
            sample_id: None,
            chrom: Some("chrX".to_string()),
            cellular_prevalence: None, // no prevalence → truncal detection fails
            outlier_prob: None,
        }];

        let err = build_compat_data_points(
            &[r0, r1],
            Some(&clusters),
            &CompatLoaderConfig {
                assign_loss_prob: true,
                global_outlier_prob: 0.01,
                loss_prob: 0.05,
                high_loss_prob: 0.4,
                ..CompatLoaderConfig::default()
            },
        )
        .expect_err("loader must reject missing cellular_prevalence for assign_loss_prob");
        assert!(err.contains("cellular_prevalence"));
    }

    #[test]
    fn assign_loss_prob_is_enabled_and_does_not_error() {
        // Phase L2: assign_loss_prob is now supported and should not error at the loader level.
        // This test verifies that the basic flow works without errors.
        let mut r0 = base_row("m0", "s0", 5);
        let mut r1 = base_row("m0", "s1", 6);
        r0.chrom = Some("chrX".to_string());
        r1.chrom = Some("chrX".to_string());

        let clusters = vec![CompatClusterRow {
            mutation_id: "m0".to_string(),
            cluster_id: "cloneA".to_string(),
            sample_id: None,
            chrom: Some("chrX".to_string()),
            cellular_prevalence: Some(0.5),
            outlier_prob: None,
        }];

        let result = build_compat_data_points(
            &[r0, r1],
            Some(&clusters),
            &CompatLoaderConfig {
                assign_loss_prob: true,
                global_outlier_prob: 0.01,
                high_loss_prob: 0.4,
                ..CompatLoaderConfig::default()
            },
        );
        // Should not error.
        assert!(result.is_ok(), "assign_loss_prob should now be supported");
    }

    #[test]
    fn ignores_cellular_prevalence_for_outlier_prob_fallback() {
        let rows = vec![base_row("m0", "s0", 5), base_row("m0", "s1", 6)];
        let clusters = vec![CompatClusterRow {
            mutation_id: "m0".to_string(),
            cluster_id: "cluster-A".to_string(),
            sample_id: None,
            chrom: None,
            cellular_prevalence: Some(0.8),
            outlier_prob: None,
        }];

        let points = build_compat_data_points(
            &rows,
            Some(&clusters),
            &CompatLoaderConfig {
                global_outlier_prob: 0.07,
                ..CompatLoaderConfig::default()
            },
        )
        .expect("loader should succeed");
        assert_eq!(points.len(), 1);

        // In PhyClone-compatible mode we do NOT infer outlier_prob from cellular_prevalence.
        // Fallback should come from global_outlier_prob.
        let expected_log = 0.07_f64.ln();
        assert!((points[0].outlier_prob - expected_log).abs() < 1e-12);
    }

    fn assert_oracle_fixture(path_str: &str) {
        let path = Path::new(path_str);
        if !path.exists() {
            eprintln!(
                "skipping oracle parity test: fixture not found at {}",
                path_str
            );
            return;
        }

        let contents = fs::read_to_string(path).expect("fixture read should succeed");
        let fixture: OracleFixture =
            serde_json::from_str(&contents).expect("fixture json should decode");

        let config = CompatLoaderConfig {
            density: parse_density(&fixture.config.density),
            precision: fixture.config.precision,
            grid_size: fixture.config.grid_size,
            global_outlier_prob: 1e-4,
            ..CompatLoaderConfig::default()
        };

        let rows: Vec<CompatInputRow> = fixture
            .rows
            .into_iter()
            .map(|row| CompatInputRow {
                mutation_id: row.mutation_id,
                sample_id: row.sample_id,
                ref_counts: row.ref_counts,
                alt_counts: row.alt_counts,
                major_cn: row.major_cn,
                minor_cn: row.minor_cn,
                normal_cn: row.normal_cn,
                tumour_content: row.tumour_content,
                error_rate: row.error_rate,
                chrom: row.chrom,
                loss_prob: row.loss_prob,
                cluster_id: row.cluster_id,
                outlier_prob: row.outlier_prob,
            })
            .collect();

        let cluster_rows: Vec<CompatClusterRow> = fixture
            .cluster_rows
            .into_iter()
            .map(|row| CompatClusterRow {
                mutation_id: row.mutation_id,
                cluster_id: row.cluster_id,
                sample_id: None,
                chrom: None,
                cellular_prevalence: row.cellular_prevalence,
                outlier_prob: row.outlier_prob,
            })
            .collect();

        let expected_grid = fixture.oracle.ccf_grid;
        let actual_grid = build_ccf_grid(CompatGridConfig {
            grid_size: config.grid_size,
        });
        assert_eq!(actual_grid.len(), expected_grid.len());
        for (idx, (actual, expected)) in actual_grid.iter().zip(expected_grid.iter()).enumerate() {
            assert_close(
                *actual,
                *expected,
                1e-12,
                &format!("grid mismatch at {}", idx),
            );
        }

        let actual_points = build_compat_data_points(
            &rows,
            if cluster_rows.is_empty() {
                None
            } else {
                Some(cluster_rows.as_slice())
            },
            &config,
        )
        .expect("compat loader should succeed");
        let expected_points = fixture.oracle.datapoints;
        assert_eq!(actual_points.len(), expected_points.len());

        for (actual, expected) in actual_points.iter().zip(expected_points.iter()) {
            let actual_name = match &actual.name {
                super::CompatDataPointName::Str(value) => value,
                _ => panic!("expected string datapoint name"),
            };
            assert_eq!(actual_name, &expected.name);
            assert_eq!(actual.size, expected.size);
            assert_eq!(actual.value.len(), expected.value.len());

            for (sample_idx, (actual_sample, expected_sample)) in
                actual.value.iter().zip(expected.value.iter()).enumerate()
            {
                assert_eq!(actual_sample.len(), expected_sample.len());
                for (grid_idx, (actual_ll, expected_ll)) in
                    actual_sample.iter().zip(expected_sample.iter()).enumerate()
                {
                    assert_close(
                        *actual_ll,
                        *expected_ll,
                        1e-8,
                        &format!("value mismatch sample={} grid={}", sample_idx, grid_idx),
                    );
                }
            }

            if let Some(expected_outlier_log) = expected.outlier_prob_log {
                assert_close(
                    actual.outlier_prob,
                    expected_outlier_log,
                    1e-8,
                    "outlier_prob mismatch",
                );
            }
            if let Some(expected_outlier_not_log) = expected.outlier_prob_not_log {
                assert_close(
                    actual.outlier_prob_not,
                    expected_outlier_not_log,
                    1e-8,
                    "outlier_prob_not mismatch",
                );
            }
        }
    }

    #[test]
    fn matches_phyclone_oracle_fixture_beta_clustered() {
        assert_oracle_fixture(ORACLE_FIXTURE_BETA_CLUSTERED);
    }

    #[test]
    fn matches_phyclone_oracle_fixture_beta_multi_mutation_one_cluster() {
        assert_oracle_fixture(ORACLE_FIXTURE_BETA_MULTI_MUTATION_ONE_CLUSTER);
    }

    #[test]
    fn matches_phyclone_oracle_fixture_binomial_clustered() {
        assert_oracle_fixture(ORACLE_FIXTURE_BINOMIAL_CLUSTERED);
    }

    #[test]
    fn matches_phyclone_oracle_fixture_beta_outlier_split() {
        assert_oracle_fixture(ORACLE_FIXTURE_BETA_OUTLIER_SPLIT);
    }

    #[test]
    fn matches_phyclone_oracle_fixture_beta_cn_purity() {
        assert_oracle_fixture(ORACLE_FIXTURE_BETA_CN_PURITY);
    }

    #[test]
    fn matches_phyclone_oracle_fixture_beta_precision_50() {
        assert_oracle_fixture(ORACLE_FIXTURE_BETA_PRECISION_50);
    }

    #[test]
    fn truncal_chrom_array_counts_unique_mutations_not_samples() {
        let rows = vec![
            super::ClusterAssignmentRow {
                mutation_id: "m0".to_string(),
                cluster_id: "c0".to_string(),
                sample_id: "s0".to_string(),
                chrom: "chr1".to_string(),
                cellular_prevalence: Some(0.7),
            },
            super::ClusterAssignmentRow {
                mutation_id: "m0".to_string(),
                cluster_id: "c0".to_string(),
                sample_id: "s1".to_string(),
                chrom: "chr1".to_string(),
                cellular_prevalence: Some(0.6),
            },
            super::ClusterAssignmentRow {
                mutation_id: "m1".to_string(),
                cluster_id: "c0".to_string(),
                sample_id: "s0".to_string(),
                chrom: "chr2".to_string(),
                cellular_prevalence: Some(0.5),
            },
            super::ClusterAssignmentRow {
                mutation_id: "m1".to_string(),
                cluster_id: "c0".to_string(),
                sample_id: "s1".to_string(),
                chrom: "chr2".to_string(),
                cellular_prevalence: Some(0.4),
            },
        ];

        let chrom_arr = super::get_truncal_chrom_array(&rows, "c0");
        assert_eq!(chrom_arr.len(), 2);
        let chrom_set: BTreeSet<String> = chrom_arr.into_iter().collect();
        assert_eq!(chrom_set.len(), 2);
        assert!(chrom_set.contains("chr1"));
        assert!(chrom_set.contains("chr2"));
    }

    #[test]
    fn resample_without_replacement_returns_unique_items() {
        let source: Vec<i32> = (0..10).collect();
        let mut rng = StdRng::seed_from_u64(7);

        let sampled = resample_without_replacement(&source, 6, &mut rng);
        assert_eq!(sampled.len(), 6);
        let unique_count = sampled.iter().collect::<BTreeSet<_>>().len();
        assert_eq!(unique_count, sampled.len());
        assert!(sampled.iter().all(|v| source.contains(v)));
    }

    #[test]
    fn resample_without_replacement_resizes_population_when_target_exceeds_source() {
        let source = vec!["a", "b", "c"];
        let mut rng = StdRng::seed_from_u64(11);

        let sampled = resample_without_replacement(&source, 10, &mut rng);
        assert_eq!(sampled.len(), 10);

        let mut counts = std::collections::BTreeMap::new();
        for value in sampled {
            *counts.entry(value).or_insert(0) += 1;
        }

        // np.resize([a,b,c], 10) => [a,b,c,a,b,c,a,b,c,a]
        // sampling all elements without replacement preserves this multiset.
        assert_eq!(counts.get("a"), Some(&4));
        assert_eq!(counts.get("b"), Some(&3));
        assert_eq!(counts.get("c"), Some(&3));
    }
}
