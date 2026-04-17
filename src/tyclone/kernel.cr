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

  struct PcvMcmcConfig
    num_iters : Int32
    burnin : Int32
    thin : Int32
    num_clusters : Int32
    alpha : Float64
    alpha_prior_shape : Float64
    alpha_prior_rate : Float64
    init_method : UInt8
    base_measure_alpha : Float64
    base_measure_beta : Float64
    mh_step_size : Float64
    mh_precision_step : Float64
    mh_precision_proposal_precision : Float64
    precision : Float64
    density : UInt8
    use_seed : UInt8
    seed : UInt64
    print_freq : Int32
  end

  type PcvResult = Void
  type PcvError = Void

  fun pcv_fit = pcv_fit(config : PcvConfig*, rows : PcvRow*, rows_len : LibC::SizeT, num_mutations : LibC::SizeT, num_samples : LibC::SizeT, out_result : PcvResult**, out_error : PcvError**) : Int32
  fun pcv_fit_with_init = pcv_fit_with_init(config : PcvConfig*, rows : PcvRow*, rows_len : LibC::SizeT, num_mutations : LibC::SizeT, num_samples : LibC::SizeT, compat_pi : Float64*, compat_pi_len : LibC::SizeT, compat_theta : Float64*, compat_theta_len : LibC::SizeT, compat_z : Float64*, compat_z_len : LibC::SizeT, out_result : PcvResult**, out_error : PcvError**) : Int32
  fun pcv_fit_mcmc = pcv_fit_mcmc(config : PcvMcmcConfig*, rows : PcvRow*, rows_len : LibC::SizeT, num_mutations : LibC::SizeT, num_samples : LibC::SizeT, out_result : PcvResult**, out_error : PcvError**) : Int32
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
  fun pcv_error_free = pcv_error_free(err : PcvError*) : Nil
end

module Tyclone
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

    def num_saved_trace_samples : Int32
      LibPcv.pcv_result_num_saved_trace_samples(@ptr).to_i32
    end

    def mutation_cluster_ids : Slice(Int32)
      ptr = LibPcv.pcv_result_mutation_cluster_ids(@ptr)
      Slice.new(ptr, num_mutations)
    end

    def mutation_cluster_probs : Slice(Float64)
      ptr = LibPcv.pcv_result_mutation_cluster_probs(@ptr)
      Slice.new(ptr, num_mutations)
    end

    def mutation_sample_prevalence : Slice(Float64)
      ptr = LibPcv.pcv_result_mutation_sample_prevalence(@ptr)
      Slice.new(ptr, num_mutations * num_samples)
    end

    def mutation_sample_prevalence_std : Slice(Float64)
      ptr = LibPcv.pcv_result_mutation_sample_prevalence_std(@ptr)
      Slice.new(ptr, num_mutations * num_samples)
    end

    def saved_mutation_sample_prevalence : Slice(Float64)
      ptr = LibPcv.pcv_result_saved_mutation_sample_prevalence(@ptr)
      Slice.new(ptr, num_saved_trace_samples * num_mutations * num_samples)
    end

    def saved_precision_trace : Slice(Float64)
      ptr = LibPcv.pcv_result_saved_precision_trace(@ptr)
      Slice.new(ptr, num_saved_trace_samples)
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
    private record CompatInitBuffers, pi : Array(Float64), theta : Array(Float64), z : Array(Float64)

    private PYTHON_INIT_CODE = {{ read_file("#{__DIR__}/python_init.py") }}

    private def self.json_to_f64(value : JSON::Any) : Float64
      raw = value.raw
      if float_v = raw.as?(Float64)
        float_v
      elsif int_v = raw.as?(Int64)
        int_v.to_f64
      else
        raise KernelError.new("--python-compatible: expected numeric value, got #{raw.class}")
      end
    end

    private def self.run_python_init(config : Config, effective_seed : UInt64, num_mutations : Int32, num_samples : Int32) : CompatInitBuffers
      input_json = "{\"seed\":#{effective_seed},\"num_restarts\":#{config.num_restarts},\"num_clusters\":#{config.num_clusters},\"num_mutations\":#{num_mutations},\"num_samples\":#{num_samples},\"num_grid_points\":#{config.num_grid_points}}"
      proc = Process.new("python3", ["-c", PYTHON_INIT_CODE],
        input: :pipe,
        output: :pipe,
        error: STDERR
      )
      proc.input.print(input_json)
      proc.input.close
      json_str = proc.output.gets_to_end
      proc.output.close
      status = proc.wait
      unless status.success?
        raise KernelError.new("--python-compatible: python3 exited with code #{status.exit_code}")
      end
      raise KernelError.new("--python-compatible: python3 produced no output") if json_str.empty?

      doc = JSON.parse(json_str)
      restarts = doc.as_h["restarts"]?.try(&.as_a) || raise KernelError.new("--python-compatible: invalid output (missing restarts)")
      if restarts.size != config.num_restarts
        raise KernelError.new("--python-compatible: expected #{config.num_restarts} restart(s), got #{restarts.size}")
      end

      expected_pi_per_restart = config.num_clusters
      expected_theta_per_restart = config.num_clusters * num_samples * config.num_grid_points
      expected_z_per_restart = num_mutations * config.num_clusters

      pi = Array(Float64).new(config.num_restarts * expected_pi_per_restart)
      theta = Array(Float64).new(config.num_restarts * expected_theta_per_restart)
      z = Array(Float64).new(config.num_restarts * expected_z_per_restart)

      restarts.each_with_index do |restart_any, restart_index|
        restart = restart_any.as_h
        pi_values = restart["pi"]?.try(&.as_a) || raise KernelError.new("--python-compatible: restart #{restart_index} missing pi")
        theta_values = restart["theta"]?.try(&.as_a) || raise KernelError.new("--python-compatible: restart #{restart_index} missing theta")
        z_values = restart["z"]?.try(&.as_a) || raise KernelError.new("--python-compatible: restart #{restart_index} missing z")

        if pi_values.size != expected_pi_per_restart
          raise KernelError.new("--python-compatible: restart #{restart_index} pi length mismatch")
        end
        if theta_values.size != expected_theta_per_restart
          raise KernelError.new("--python-compatible: restart #{restart_index} theta length mismatch")
        end
        if z_values.size != expected_z_per_restart
          raise KernelError.new("--python-compatible: restart #{restart_index} z length mismatch")
        end

        pi_values.each { |v| pi << json_to_f64(v) }
        theta_values.each { |v| theta << json_to_f64(v) }
        z_values.each { |v| z << json_to_f64(v) }
      end

      CompatInitBuffers.new(pi, theta, z)
    end

    def self.fit(config : Config, rows : Array(LibPcv::PcvRow), num_mutations : Int32, num_samples : Int32) : KernelResult
      effective_seed = config.seed
      if config.python_compatible? && effective_seed.nil?
        effective_seed = Random::Secure.rand(UInt64::MAX)
      end

      cfg = LibPcv::PcvConfig.new(
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
        density: (config.density == Density::Binomial ? 0_u8 : 1_u8),
        use_seed: effective_seed.nil? ? 0_u8 : 1_u8,
        seed: effective_seed || 0_u64
      )

      result_ptr = Pointer(LibPcv::PcvResult).null
      error_ptr = Pointer(LibPcv::PcvError).null

      rc = if config.python_compatible?
             seed_for_python = effective_seed || raise KernelError.new("--python-compatible: seed resolution failed")
             compat = run_python_init(config, seed_for_python, num_mutations, num_samples)
             LibPcv.pcv_fit_with_init(
               pointerof(cfg),
               rows.to_unsafe,
               rows.size,
               num_mutations,
               num_samples,
               compat.pi.to_unsafe,
               compat.pi.size,
               compat.theta.to_unsafe,
               compat.theta.size,
               compat.z.to_unsafe,
               compat.z.size,
               pointerof(result_ptr),
               pointerof(error_ptr)
             )
           else
             LibPcv.pcv_fit(
               pointerof(cfg),
               rows.to_unsafe,
               rows.size,
               num_mutations,
               num_samples,
               pointerof(result_ptr),
               pointerof(error_ptr)
             )
           end

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

    def self.fit_mcmc(config : Config, rows : Array(LibPcv::PcvRow), num_mutations : Int32, num_samples : Int32) : KernelResult
      cfg = LibPcv::PcvMcmcConfig.new(
        num_iters: config.num_iters,
        burnin: config.burnin,
        thin: config.thin,
        num_clusters: config.num_clusters,
        alpha: config.alpha,
        alpha_prior_shape: config.alpha_prior_shape,
        alpha_prior_rate: config.alpha_prior_rate,
        init_method: (config.init_method == "connected" ? 1_u8 : 0_u8),
        base_measure_alpha: config.base_measure_alpha,
        base_measure_beta: config.base_measure_beta,
        mh_step_size: config.mh_step_size,
        mh_precision_step: config.mh_precision_step,
        mh_precision_proposal_precision: config.mh_precision_proposal_precision,
        precision: config.precision,
        density: (config.density == Density::Binomial ? 0_u8 : 1_u8),
        use_seed: config.seed.nil? ? 0_u8 : 1_u8,
        seed: config.seed || 0_u64,
        print_freq: config.print_freq
      )

      result_ptr = Pointer(LibPcv::PcvResult).null
      error_ptr = Pointer(LibPcv::PcvError).null

      rc = LibPcv.pcv_fit_mcmc(
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
