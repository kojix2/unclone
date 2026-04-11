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
    convergence_threshold : Float64
    mix_weight_prior : Float64
    precision : Float64
    density : UInt8
    use_seed : UInt8
    seed : UInt64
  end

  type PcvResult = Void
  type PcvError = Void

  fun pcv_fit = pcv_fit(config : PcvConfig*, rows : PcvRow*, rows_len : LibC::SizeT, num_mutations : LibC::SizeT, num_samples : LibC::SizeT, out_result : PcvResult**, out_error : PcvError**) : Int32
  fun pcv_result_num_mutations = pcv_result_num_mutations(result : PcvResult*) : LibC::SizeT
  fun pcv_result_num_samples = pcv_result_num_samples(result : PcvResult*) : LibC::SizeT
  fun pcv_result_num_clusters = pcv_result_num_clusters(result : PcvResult*) : LibC::SizeT
  fun pcv_result_mutation_cluster_ids = pcv_result_mutation_cluster_ids(result : PcvResult*) : Int32*
  fun pcv_result_mutation_cluster_probs = pcv_result_mutation_cluster_probs(result : PcvResult*) : Float64*
  fun pcv_result_cluster_sample_prevalence = pcv_result_cluster_sample_prevalence(result : PcvResult*) : Float64*
  fun pcv_result_cluster_sample_prevalence_std = pcv_result_cluster_sample_prevalence_std(result : PcvResult*) : Float64*
  fun pcv_result_free = pcv_result_free(result : PcvResult*) : Nil
  fun pcv_error_message = pcv_error_message(err : PcvError*) : UInt8*
  fun pcv_error_free = pcv_error_free(err : PcvError*) : Nil
end

module Toyclone
  class KernelResult
    def initialize(@ptr : LibPcv::PcvResult*)
    end

    def num_mutations : Int32
      LibPcv.pcv_result_num_mutations(@ptr).to_i32
    end

    def num_samples : Int32
      LibPcv.pcv_result_num_samples(@ptr).to_i32
    end

    def num_clusters : Int32
      LibPcv.pcv_result_num_clusters(@ptr).to_i32
    end

    def mutation_cluster_ids : Slice(Int32)
      ptr = LibPcv.pcv_result_mutation_cluster_ids(@ptr)
      Slice.new(ptr, num_mutations)
    end

    def mutation_cluster_probs : Slice(Float64)
      ptr = LibPcv.pcv_result_mutation_cluster_probs(@ptr)
      Slice.new(ptr, num_mutations)
    end

    def cluster_sample_prevalence : Slice(Float64)
      ptr = LibPcv.pcv_result_cluster_sample_prevalence(@ptr)
      Slice.new(ptr, num_clusters * num_samples)
    end

    def cluster_sample_prevalence_std : Slice(Float64)
      ptr = LibPcv.pcv_result_cluster_sample_prevalence_std(@ptr)
      Slice.new(ptr, num_clusters * num_samples)
    end

    def free
      LibPcv.pcv_result_free(@ptr)
    end
  end

  module Kernel
    def self.fit(config : Config, rows : Array(LibPcv::PcvRow), num_mutations : Int32, num_samples : Int32) : KernelResult
      cfg = LibPcv::PcvConfig.new(
        num_clusters: config.num_clusters,
        num_grid_points: config.num_grid_points,
        num_restarts: config.num_restarts,
        max_iters: config.max_iters,
        print_freq: config.print_freq,
        kernel_threads: config.kernel_threads,
        convergence_threshold: config.convergence_threshold,
        mix_weight_prior: config.mix_weight_prior,
        precision: config.precision,
        density: (config.density == Density::Binomial ? 0_u8 : 1_u8),
        use_seed: config.seed.nil? ? 0_u8 : 1_u8,
        seed: config.seed || 0_u64
      )

      result_ptr = Pointer(LibPcv::PcvResult).null
      error_ptr = Pointer(LibPcv::PcvError).null

      rc = LibPcv.pcv_fit(
        pointerof(cfg),
        rows.to_unsafe,
        rows.size,
        num_mutations,
        num_samples,
        pointerof(result_ptr),
        pointerof(error_ptr)
      )

      if rc != 0
        message = "Unknown kernel error"
        unless error_ptr.null?
          message_ptr = LibPcv.pcv_error_message(error_ptr)
          message = String.new(message_ptr) unless message_ptr.null?
          LibPcv.pcv_error_free(error_ptr)
        end
        raise KernelError.new(message)
      end

      KernelResult.new(result_ptr)
    end
  end
end
