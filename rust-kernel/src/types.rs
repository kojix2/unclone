use std::ffi::CString;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PcvRow {
    pub mutation_index: i32,
    pub sample_index: i32,
    pub ref_counts: i32,
    pub alt_counts: i32,
    pub major_cn: i32,
    pub minor_cn: i32,
    pub normal_cn: i32,
    pub tumour_content: f64,
    pub error_rate: f64,
}

#[repr(C)]
pub struct PcvConfig {
    pub num_clusters: i32,
    pub num_grid_points: i32,
    pub num_restarts: i32,
    pub max_iters: i32,
    pub print_freq: i32,
    pub kernel_threads: i32,
    pub convergence_threshold: f64,
    pub mix_weight_prior: f64,
    pub precision: f64,
    pub density: u8,
    pub use_seed: u8,
    pub seed: u64,
}

pub struct PcvResult {
    pub num_mutations: usize,
    pub num_samples: usize,
    pub num_clusters: usize,
    pub mutation_cluster_ids: Vec<i32>,
    pub mutation_cluster_probs: Vec<f64>,
    pub mutation_sample_prevalence: Vec<f64>,
    pub mutation_sample_prevalence_std: Vec<f64>,
    pub cluster_sample_prevalence: Vec<f64>,
    pub cluster_sample_prevalence_std: Vec<f64>,
}

pub struct PcvError {
    pub message: CString,
}

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

// ── MCMC types ─────────────────────────────────────────────────────────────

/// C-ABI configuration passed from Crystal for the MCMC engine.
#[repr(C)]
pub struct PcvMcmcConfig {
    /// Number of post-burn-in MCMC iterations to keep.
    pub num_iters: i32,
    /// Number of burn-in iterations to discard.
    pub burnin: i32,
    /// Thinning factor: retain every `thin`-th sample.
    pub thin: i32,
    /// Maximum number of clusters used in post-processing / output summarisation.
    pub num_clusters: i32,
    /// CRP concentration parameter α.
    pub alpha: f64,
    /// Gamma prior shape for α (Escobar & West 1995).
    pub alpha_prior_shape: f64,
    /// Gamma prior rate for α.
    pub alpha_prior_rate: f64,
    /// Initialization mode: 0 = disconnected, 1 = connected.
    pub init_method: u8,
    /// Beta base measure alpha parameter.
    pub base_measure_alpha: f64,
    /// Beta base measure beta parameter.
    pub base_measure_beta: f64,
    /// Step size for the per-cluster atom MH proposal (log-scale).
    pub mh_step_size: f64,
    /// Step size for the precision MH proposal (0 = fixed precision).
    pub mh_precision_step: f64,
    /// Proposal precision parameter for the Gamma precision proposal.
    pub mh_precision_proposal_precision: f64,
    /// Beta-binomial precision (used when density == BetaBinomial).
    pub precision: f64,
    /// Density code: 0 = Binomial, 1 = BetaBinomial.
    pub density: u8,
    /// Whether to seed the RNG.
    pub use_seed: u8,
    /// RNG seed value (ignored when use_seed == 0).
    pub seed: u64,
    /// Print progress every N iterations (0 = silent).
    pub print_freq: i32,
}

/// Per-cluster atom: the cellular prevalence φ_{k,s} for each sample s.
#[derive(Debug, Clone)]
pub struct ClusterAtom {
    /// Length = num_samples.
    pub phi: Vec<f64>,
}

/// Full DP partition state at one MCMC step.
#[derive(Debug, Clone)]
pub struct DpState {
    /// `cluster_id[m]` = which cluster mutation m belongs to (0-based, compact).
    pub cluster_id: Vec<usize>,
    /// Active cluster atoms, indexed by compact cluster id.
    pub atoms: Vec<ClusterAtom>,
    /// Current CRP concentration parameter.
    pub alpha: f64,
    /// Current beta-binomial precision (if applicable).
    pub precision: f64,
}

/// Collected MCMC samples used for posterior summarisation.
#[derive(Debug, Clone, Default)]
pub struct McmcTrace {
    /// Co-cluster matrix accumulator: entry (i,j) counts how often mutation i
    /// and mutation j were assigned to the same cluster across saved iterations.
    /// Flat row-major, length = num_mutations × num_mutations.
    pub co_cluster_counts: Vec<u32>,
    /// Number of saved samples (post-burn-in, post-thinning).
    pub num_samples: usize,
    /// Per-mutation, per-sample CCF sum across saved samples (for mean).
    /// Flat: index = mutation_index * num_samples_data + sample_index.
    pub ccf_sum: Vec<f64>,
    /// Per-mutation, per-sample CCF sum-of-squares (for std).
    pub ccf_sum_sq: Vec<f64>,
    /// Sum of saved precision values for post-processing.
    pub precision_sum: f64,
}
