#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConvolutionCacheStats {
    pub hits: usize,
    pub misses: usize,
    pub entries: usize,
}

#[derive(Clone, Debug)]
pub struct ConvolutionCache {
    max_entries: usize,
    hits: usize,
    misses: usize,
    order: VecDeque<ConvolutionKey>,
    map: HashMap<ConvolutionKey, Vec<f64>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ConvolutionKey {
    left: Vec<u64>,
    right: Vec<u64>,
}

impl ConvolutionCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries,
            hits: 0,
            misses: 0,
            order: VecDeque::new(),
            map: HashMap::new(),
        }
    }

    pub fn convolve_two_children(&mut self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let key = ConvolutionKey::from_slices(a, b);

        if let Some(value) = self.map.get(&key) {
            self.hits += 1;
            return value.clone();
        }

        self.misses += 1;
        let result = convolve_two_children_1d(a, b);

        if self.max_entries == 0 {
            return result;
        }

        self.order.push_back(key.clone());
        self.map.insert(key, result.clone());

        while self.map.len() > self.max_entries {
            if let Some(oldest) = self.order.pop_front() {
                self.map.remove(&oldest);
            } else {
                break;
            }
        }

        result
    }

    pub fn stats(&self) -> ConvolutionCacheStats {
        ConvolutionCacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.map.len(),
        }
    }
}

impl ConvolutionKey {
    fn from_slices(a: &[f64], b: &[f64]) -> Self {
        Self {
            left: a.iter().map(|v| v.to_bits()).collect(),
            right: b.iter().map(|v| v.to_bits()).collect(),
        }
    }
}

pub fn compute_log_d_1d(child_log_r_values: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    compute_log_d_1d_with_cache(child_log_r_values, None)
}

pub fn compute_log_d_1d_with_cache(
    child_log_r_values: &[Vec<f64>],
    mut cache: Option<&mut ConvolutionCache>,
) -> Result<Vec<f64>, String> {
    match child_log_r_values.len() {
        0 => return Ok(vec![0.0]),
        1 => return Ok(child_log_r_values[0].clone()),
        _ => {}
    }

    let grid_size = child_log_r_values[0].len();
    if grid_size == 0 {
        return Ok(Vec::new());
    }

    for child in child_log_r_values {
        if child.len() != grid_size {
            return Err("all child log_R arrays must have the same grid size".to_string());
        }
    }

    let mut maxes = Vec::with_capacity(child_log_r_values.len());
    let mut normed = Vec::with_capacity(child_log_r_values.len());

    for child in child_log_r_values {
        let max = child.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        maxes.push(max);

        let mut arr = vec![0.0_f64; grid_size];
        for (idx, value) in child.iter().enumerate() {
            arr[idx] = (*value - max).exp();
        }
        normed.push(arr);
    }

    let mut conv = match cache.as_deref_mut() {
        Some(c) => c.convolve_two_children(&normed[0], &normed[1]),
        None => convolve_two_children_1d(&normed[0], &normed[1]),
    };

    for child in normed.iter().skip(2) {
        conv = match cache.as_deref_mut() {
            Some(c) => c.convolve_two_children(child, &conv),
            None => convolve_two_children_1d(child, &conv),
        };
    }

    let max_sum: f64 = maxes.iter().sum();
    for value in &mut conv {
        if *value <= 0.0 {
            *value = 1e-100;
        }
        *value = value.ln() + max_sum;
    }

    Ok(conv)
}

pub fn compute_log_s_1d(child_log_r_values: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    compute_log_s_1d_with_cache(child_log_r_values, None)
}

pub fn compute_log_s_1d_with_cache(
    child_log_r_values: &[Vec<f64>],
    cache: Option<&mut ConvolutionCache>,
) -> Result<Vec<f64>, String> {
    if child_log_r_values.is_empty() {
        return Ok(vec![0.0]);
    }

    let log_d = compute_log_d_1d_with_cache(child_log_r_values, cache)?;
    Ok(cumulative_log_add_exp(&log_d))
}

fn convolve_two_children_1d(a: &[f64], b: &[f64]) -> Vec<f64> {
    let grid_size = a.len();
    let mut out = vec![0.0_f64; grid_size];

    for j in 0..grid_size {
        let mut sum = 0.0;
        for i in 0..=j {
            sum += a[i] * b[j - i];
        }
        out[j] = sum;
    }

    out
}

fn cumulative_log_add_exp(values: &[f64]) -> Vec<f64> {
    let mut out = vec![f64::NEG_INFINITY; values.len()];
    let mut running = f64::NEG_INFINITY;

    for (idx, value) in values.iter().enumerate() {
        running = log_add_exp(running, *value);
        out[idx] = running;
    }

    out
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    if a.is_infinite() && a.is_sign_negative() {
        return b;
    }
    if b.is_infinite() && b.is_sign_negative() {
        return a;
    }

    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::{
        compute_log_d_1d, compute_log_d_1d_with_cache, compute_log_s_1d,
        compute_log_s_1d_with_cache, ConvolutionCache,
    };

    fn lse(values: &[f64]) -> f64 {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max.is_infinite() && max.is_sign_negative() {
            return max;
        }
        let sum: f64 = values.iter().map(|v| (*v - max).exp()).sum();
        max + sum.ln()
    }

    fn brute_force_log_d(children: &[Vec<f64>]) -> Vec<f64> {
        let grid = children[0].len();
        (0..grid)
            .map(|target| brute_force_for_target(children, target, 0, 0, 0.0))
            .collect()
    }

    fn brute_force_for_target(
        children: &[Vec<f64>],
        target: usize,
        child_idx: usize,
        used_sum: usize,
        acc: f64,
    ) -> f64 {
        if child_idx == children.len() {
            if used_sum == target {
                return acc;
            }
            return f64::NEG_INFINITY;
        }

        let mut terms = Vec::new();
        let remaining = target.saturating_sub(used_sum);
        for i in 0..=remaining {
            terms.push(brute_force_for_target(
                children,
                target,
                child_idx + 1,
                used_sum + i,
                acc + children[child_idx][i],
            ));
        }

        lse(&terms)
    }

    #[test]
    fn no_children_returns_scalar_zero_identity() {
        let actual = compute_log_d_1d(&[]).expect("no-child case should succeed");
        assert_eq!(actual, vec![0.0]);
    }

    #[test]
    fn one_child_is_passthrough() {
        let child = vec![-1.2, -0.2, -3.4];
        let actual = compute_log_d_1d(&[child.clone()]).expect("one-child case should succeed");
        assert_eq!(actual, child);
    }

    #[test]
    fn two_children_matches_bruteforce() {
        let children = vec![vec![-0.2, -1.1, -0.7], vec![-1.2, -0.3, -2.0]];

        let actual = compute_log_d_1d(&children).expect("two-child case should succeed");
        let expected = brute_force_log_d(&children);

        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= 1e-10);
        }
    }

    #[test]
    fn three_children_matches_bruteforce() {
        let children = vec![
            vec![-0.2, -1.1, -0.7],
            vec![-1.2, -0.3, -2.0],
            vec![-0.6, -0.7, -1.5],
        ];

        let actual = compute_log_d_1d(&children).expect("three-child case should succeed");
        let expected = brute_force_log_d(&children);

        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= 1e-10);
        }
    }

    #[test]
    fn errors_on_grid_size_mismatch() {
        let children = vec![vec![-1.0, -2.0], vec![-0.5, -1.2, -2.1]];
        let err = compute_log_d_1d(&children).expect_err("grid mismatch must error");
        assert!(err.contains("grid size"));
    }

    #[test]
    fn compute_log_s_no_children_returns_scalar_zero_identity() {
        let actual = compute_log_s_1d(&[]).expect("no-child case should succeed");
        assert_eq!(actual, vec![0.0]);
    }

    #[test]
    fn compute_log_s_one_child_matches_cumulative_logaddexp() {
        let children = vec![vec![-1.2, -0.2, -3.4]];

        let actual = compute_log_s_1d(&children).expect("one-child case should succeed");

        let expected0 = children[0][0];
        let expected1 = lse(&[children[0][0], children[0][1]]);
        let expected2 = lse(&[children[0][0], children[0][1], children[0][2]]);

        assert!((actual[0] - expected0).abs() <= 1e-10);
        assert!((actual[1] - expected1).abs() <= 1e-10);
        assert!((actual[2] - expected2).abs() <= 1e-10);
    }

    #[test]
    fn compute_log_s_matches_cumulative_logaddexp_of_log_d() {
        let children = vec![
            vec![-0.2, -1.1, -0.7],
            vec![-1.2, -0.3, -2.0],
            vec![-0.6, -0.7, -1.5],
        ];

        let log_d = compute_log_d_1d(&children).expect("log_d should succeed");
        let actual = compute_log_s_1d(&children).expect("log_s should succeed");

        let mut expected = Vec::new();
        for i in 0..log_d.len() {
            expected.push(lse(&log_d[0..=i]));
        }

        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() <= 1e-10);
        }
    }

    #[test]
    fn compute_log_s_errors_on_grid_size_mismatch() {
        let children = vec![vec![-1.0, -2.0], vec![-0.5, -1.2, -2.1]];
        let err = compute_log_s_1d(&children).expect_err("grid mismatch must error");
        assert!(err.contains("grid size"));
    }

    #[test]
    fn cache_tracks_hits_and_misses_for_repeated_convolution() {
        let children = vec![
            vec![-0.2, -1.1, -0.7],
            vec![-1.2, -0.3, -2.0],
            vec![-0.6, -0.7, -1.5],
        ];
        let mut cache = ConvolutionCache::new(16);

        let _ = compute_log_d_1d_with_cache(&children, Some(&mut cache))
            .expect("first call should succeed");
        let _ = compute_log_d_1d_with_cache(&children, Some(&mut cache))
            .expect("second call should succeed");

        let stats = cache.stats();
        assert!(stats.misses >= 2);
        assert!(stats.hits >= 2);
        assert!(stats.entries <= 16);
    }

    #[test]
    fn cache_eviction_respects_capacity() {
        let mut cache = ConvolutionCache::new(1);

        let children_a = vec![vec![-0.2, -1.1, -0.7], vec![-1.2, -0.3, -2.0]];
        let children_b = vec![vec![-0.3, -0.8, -1.4], vec![-0.5, -0.6, -1.1]];

        let _ = compute_log_d_1d_with_cache(&children_a, Some(&mut cache))
            .expect("call A should succeed");
        let _ = compute_log_d_1d_with_cache(&children_b, Some(&mut cache))
            .expect("call B should succeed");

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
    }

    #[test]
    fn compute_log_s_with_cache_matches_plain_version() {
        let children = vec![
            vec![-0.2, -1.1, -0.7],
            vec![-1.2, -0.3, -2.0],
            vec![-0.6, -0.7, -1.5],
        ];

        let mut cache = ConvolutionCache::new(8);
        let cached = compute_log_s_1d_with_cache(&children, Some(&mut cache))
            .expect("cached log_s should succeed");
        let plain = compute_log_s_1d(&children).expect("plain log_s should succeed");

        assert_eq!(cached.len(), plain.len());
        for (c, p) in cached.iter().zip(plain.iter()) {
            assert!((c - p).abs() <= 1e-10);
        }
    }
}
