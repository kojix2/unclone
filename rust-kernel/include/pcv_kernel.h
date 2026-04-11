#ifndef PCV_KERNEL_H
#define PCV_KERNEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  int32_t mutation_index;
  int32_t sample_index;
  int32_t ref_counts;
  int32_t alt_counts;
  int32_t major_cn;
  int32_t minor_cn;
  int32_t normal_cn;
  double tumour_content;
  double error_rate;
} PcvRow;

typedef struct {
  int32_t num_clusters;
  int32_t num_grid_points;
  int32_t num_restarts;
  int32_t max_iters;
  int32_t print_freq;
  int32_t kernel_threads;
  int32_t restart_parallelism;
  double convergence_threshold;
  double mix_weight_prior;
  double precision;
  uint8_t density;
  uint8_t use_seed;
  uint64_t seed;
} PcvConfig;

typedef struct PcvTabularResult PcvTabularResult;
typedef PcvTabularResult PcvResult;
typedef struct PcvError PcvError;

int pcv_fit(
  const PcvConfig* config,
  const PcvRow* rows,
  size_t rows_len,
  size_t num_mutations,
  size_t num_samples,
  PcvResult** out_result,
  PcvError** out_error
);

int pcv_fit_with_init(
  const PcvConfig* config,
  const PcvRow* rows,
  size_t rows_len,
  size_t num_mutations,
  size_t num_samples,
  const double* compat_pi,
  size_t compat_pi_len,
  const double* compat_theta,
  size_t compat_theta_len,
  const double* compat_z,
  size_t compat_z_len,
  PcvResult** out_result,
  PcvError** out_error
);

size_t pcv_result_num_mutations(const PcvResult* result);
size_t pcv_result_num_samples(const PcvResult* result);
size_t pcv_result_num_clusters(const PcvResult* result);
size_t pcv_result_num_saved_trace_samples(const PcvResult* result);

const int32_t* pcv_result_mutation_cluster_ids(const PcvResult* result);
const double* pcv_result_mutation_cluster_probs(const PcvResult* result);
const double* pcv_result_mutation_sample_prevalence(const PcvResult* result);
const double* pcv_result_mutation_sample_prevalence_std(const PcvResult* result);
const double* pcv_result_saved_mutation_sample_prevalence(const PcvResult* result);
const double* pcv_result_saved_precision_trace(const PcvResult* result);
const double* pcv_result_cluster_sample_prevalence(const PcvResult* result);
const double* pcv_result_cluster_sample_prevalence_std(const PcvResult* result);

void pcv_result_free(PcvResult* result);

const char* pcv_error_message(const PcvError* err);
void pcv_error_free(PcvError* err);

#endif
