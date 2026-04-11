#![allow(dead_code)]

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompatGridConfig {
    pub grid_size: usize,
}

impl Default for CompatGridConfig {
    fn default() -> Self {
        Self { grid_size: 101 }
    }
}

pub fn build_ccf_grid(config: CompatGridConfig) -> Vec<f64> {
    let n = config.grid_size.max(2);
    let step = 1.0 / ((n - 1) as f64);

    (0..n).map(|i| (i as f64) * step).collect()
}

#[cfg(test)]
mod tests {
    use super::{build_ccf_grid, CompatGridConfig};

    #[test]
    fn builds_closed_linspace_grid() {
        let grid = build_ccf_grid(CompatGridConfig { grid_size: 5 });
        assert_eq!(grid, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }
}
