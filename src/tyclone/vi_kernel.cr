require "json"

module Tyclone
  module ViKernel
    private record CompatInitBuffers, pi : Array(Float64), theta : Array(Float64), z : Array(Float64)

    private PYTHON_INIT_CODE = {{ read_file("#{__DIR__}/python_init_vi.py") }}

    private def self.resolve_python_executable : String
      ENV["TYCLONE_PYTHON"]? || "python3"
    end

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

    private def self.run_python_init(config : ViConfig, effective_seed : UInt64, num_mutations : Int32, num_samples : Int32) : CompatInitBuffers
      input_json = "{\"seed\":#{effective_seed},\"num_restarts\":#{config.num_restarts},\"num_clusters\":#{config.num_clusters},\"num_mutations\":#{num_mutations},\"num_samples\":#{num_samples},\"num_grid_points\":#{config.num_grid_points}}"
      python_executable = resolve_python_executable
      proc = Process.new(python_executable, ["-c", PYTHON_INIT_CODE],
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
        raise KernelError.new("--python-compatible: #{python_executable} exited with code #{status.exit_code}")
      end
      raise KernelError.new("--python-compatible: #{python_executable} produced no output") if json_str.empty?

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

    def self.fit(config : ViConfig, rows : Array(KernelAbi::Row), num_mutations : Int32, num_samples : Int32) : PcvTabularResult
      effective_seed = config.seed
      if config.python_compatible? && effective_seed.nil?
        effective_seed = Random::Secure.rand(UInt64::MAX)
      end

      cfg = KernelAbi.build_vi_config(config, effective_seed)

      result_ptr = Pointer(KernelAbi::Result).null
      error_ptr = Pointer(KernelAbi::Error).null

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

      Kernel.handle_result(result_ptr, error_ptr, rc)
    end
  end
end
