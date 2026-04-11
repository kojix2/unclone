#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Density {
    Binomial,
    BetaBinomial,
}

impl TryFrom<u8> for Density {
    type Error = String;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Binomial),
            1 => Ok(Self::BetaBinomial),
            _ => Err(format!("unknown density code: {value}")),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MajorCnPrior {
    pub cn: Vec<[i32; 3]>,
    pub mu: Vec<[f64; 3]>,
    pub log_pi: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampleDataPoint {
    pub a: i32,
    pub b: i32,
    pub cn: Vec<[i32; 3]>,
    pub mu: Vec<[f64; 3]>,
    pub log_pi: Vec<f64>,
    pub t: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogLikelihoodTensor {
    pub num_mutations: usize,
    pub num_samples: usize,
    pub num_grid_points: usize,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DataPreprocessor {
    pub theta_update_data: Vec<f64>,
    pub z_update_data: Vec<f64>,
    pub theta_update_shape: (usize, usize),
    pub z_update_shape: usize,
    pub use_parallel: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Priors {
    pub pi: Vec<f64>,
    pub theta: Vec<f64>,
    pub log_theta: Vec<f64>,
    pub pi_log_gamma: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariationalParameters {
    pub pi: Vec<f64>,
    pub theta: Vec<f64>,
    pub z: Vec<f64>,
    pub num_clusters: usize,
    pub num_data_points: usize,
    pub num_dims: usize,
    pub num_grid_points: usize,
}
