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
    pub restart_parallelism: i32,
    pub convergence_threshold: f64,
    pub mix_weight_prior: f64,
    pub precision: f64,
    pub density: u8,
    pub use_seed: u8,
    pub seed: u64,
}

pub struct PcvTabularResult {
    pub num_mutations: usize,
    pub num_samples: usize,
    pub num_clusters: usize,
    pub num_saved_trace_samples: usize,
    pub mutation_cluster_ids: Vec<i32>,
    pub mutation_cluster_probs: Vec<f64>,
    pub mutation_sample_prevalence: Vec<f64>,
    pub mutation_sample_prevalence_std: Vec<f64>,
    pub saved_mutation_sample_prevalence: Vec<f64>,
    pub saved_precision_trace: Vec<f64>,
    pub cluster_sample_prevalence: Vec<f64>,
    pub cluster_sample_prevalence_std: Vec<f64>,
}

pub type PcvResult = PcvTabularResult;

pub struct PcvError {
    pub message: CString,
}
