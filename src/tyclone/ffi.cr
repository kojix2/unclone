require "json"

# Linker inputs are supplied by the Makefile so this FFI block intentionally
# omits a Crystal @[Link] annotation.
lib LibPcv
  struct PcvRow
    mutation_index : Int32
    sample_index : Int32
    ref_counts : Int32
    alt_counts : Int32
    major_cn : Int32
    minor_cn : Int32
    normal_cn : Int32
    tumour_content : Float64
    error_rate : Float64
  end

  struct PcvConfig
    num_clusters : Int32
    num_grid_points : Int32
    num_restarts : Int32
    max_iters : Int32
    print_freq : Int32
    kernel_threads : Int32
    restart_parallelism : Int32
    convergence_threshold : Float64
    mix_weight_prior : Float64
    precision : Float64
    density : UInt8
    use_seed : UInt8
    seed : UInt64
  end

  type PcvTabularResult = Void
  alias PcvResult = PcvTabularResult
  type PcvError = Void

  fun pcv_fit = pcv_fit(config : PcvConfig*, rows : PcvRow*, rows_len : LibC::SizeT, num_mutations : LibC::SizeT, num_samples : LibC::SizeT, out_result : PcvResult**, out_error : PcvError**) : Int32
  fun pcv_fit_with_init = pcv_fit_with_init(config : PcvConfig*, rows : PcvRow*, rows_len : LibC::SizeT, num_mutations : LibC::SizeT, num_samples : LibC::SizeT, compat_pi : Float64*, compat_pi_len : LibC::SizeT, compat_theta : Float64*, compat_theta_len : LibC::SizeT, compat_z : Float64*, compat_z_len : LibC::SizeT, out_result : PcvResult**, out_error : PcvError**) : Int32
  fun pcv_phyclone_generate_trace = pcv_phyclone_generate_trace(cluster_json : UInt8*, num_chains : Int32, num_iters : Int32, use_seed : UInt8, seed : UInt64, out_json : UInt8**, out_error : PcvError**) : Int32
  fun pcv_result_num_mutations = pcv_result_num_mutations(result : PcvResult*) : LibC::SizeT
  fun pcv_result_num_samples = pcv_result_num_samples(result : PcvResult*) : LibC::SizeT
  fun pcv_result_num_clusters = pcv_result_num_clusters(result : PcvResult*) : LibC::SizeT
  fun pcv_result_num_saved_trace_samples = pcv_result_num_saved_trace_samples(result : PcvResult*) : LibC::SizeT
  fun pcv_result_mutation_cluster_ids = pcv_result_mutation_cluster_ids(result : PcvResult*) : Int32*
  fun pcv_result_mutation_cluster_probs = pcv_result_mutation_cluster_probs(result : PcvResult*) : Float64*
  fun pcv_result_mutation_sample_prevalence = pcv_result_mutation_sample_prevalence(result : PcvResult*) : Float64*
  fun pcv_result_mutation_sample_prevalence_std = pcv_result_mutation_sample_prevalence_std(result : PcvResult*) : Float64*
  fun pcv_result_saved_mutation_sample_prevalence = pcv_result_saved_mutation_sample_prevalence(result : PcvResult*) : Float64*
  fun pcv_result_saved_precision_trace = pcv_result_saved_precision_trace(result : PcvResult*) : Float64*
  fun pcv_result_cluster_sample_prevalence = pcv_result_cluster_sample_prevalence(result : PcvResult*) : Float64*
  fun pcv_result_cluster_sample_prevalence_std = pcv_result_cluster_sample_prevalence_std(result : PcvResult*) : Float64*
  fun pcv_result_free = pcv_result_free(result : PcvResult*) : Nil
  fun pcv_error_message = pcv_error_message(err : PcvError*) : UInt8*
  fun pcv_string_free = pcv_string_free(value : UInt8*) : Nil
  fun pcv_error_free = pcv_error_free(err : PcvError*) : Nil
end

module Tyclone
  module KernelAbi
    alias Row = LibPcv::PcvRow
    alias ViConfig = LibPcv::PcvConfig
    alias TabularResult = LibPcv::PcvTabularResult
    alias Result = TabularResult
    alias Error = LibPcv::PcvError

    def self.build_rows(rows : Array(IndexedRow)) : Array(Row)
      rows.map do |row|
        Row.new(
          mutation_index: row.mutation_index,
          sample_index: row.sample_index,
          ref_counts: row.ref_counts,
          alt_counts: row.alt_counts,
          major_cn: row.major_cn,
          minor_cn: row.minor_cn,
          normal_cn: row.normal_cn,
          tumour_content: row.tumour_content,
          error_rate: row.error_rate
        )
      end
    end

    def self.build_vi_config(config : Tyclone::ViConfig, effective_seed : UInt64?) : ViConfig
      ViConfig.new(
        num_clusters: config.num_clusters,
        num_grid_points: config.num_grid_points,
        num_restarts: config.num_restarts,
        max_iters: config.max_iters,
        print_freq: config.print_freq,
        kernel_threads: config.kernel_threads,
        restart_parallelism: config.restart_parallelism,
        convergence_threshold: config.convergence_threshold,
        mix_weight_prior: config.mix_weight_prior,
        precision: config.precision,
        density: density_code(config.density),
        use_seed: seed_flag(effective_seed),
        seed: effective_seed || 0_u64
      )
    end

    private def self.density_code(density : Tyclone::Density) : UInt8
      density == Tyclone::Density::Binomial ? 0_u8 : 1_u8
    end

    private def self.seed_flag(effective_seed : UInt64?) : UInt8
      effective_seed.nil? ? 0_u8 : 1_u8
    end
  end
end
