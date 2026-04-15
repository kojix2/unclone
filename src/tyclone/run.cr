require "time"

module Tyclone
  struct CliProfile
    property input_read_ms : Float64 = 0.0
    property sanitize_ms : Float64 = 0.0
    property indexing_ms : Float64 = 0.0
    property kernel_ms : Float64 = 0.0
    property result_build_ms : Float64 = 0.0
    property output_write_ms : Float64 = 0.0
    property trace_dump_ms : Float64 = 0.0
    property total_ms : Float64 = 0.0

    def print_summary
      STDERR.puts(
        "[tyclone-cli-profile] input_read_ms=#{@input_read_ms} sanitize_ms=#{@sanitize_ms} indexing_ms=#{@indexing_ms} kernel_ms=#{@kernel_ms} result_build_ms=#{@result_build_ms} output_write_ms=#{@output_write_ms} trace_dump_ms=#{@trace_dump_ms} total_ms=#{@total_ms}"
      )
    end
  end

  module Run
    private def self.dump_mcmc_trace(dir : String, indexed : IndexedRows, result : KernelResult)
      Dir.mkdir_p(dir)

      samples = result.num_saved_trace_samples
      return if samples <= 0

      sample_count = indexed.num_samples
      mutation_count = indexed.num_mutations
      trace = result.saved_mutation_sample_prevalence

      indexed.sample_ids.each_with_index do |sample_id, sample_index|
        path = File.join(dir, "#{sample_id}.cellular_prevalence.tsv")
        File.open(path, "w") do |file|
          file.puts indexed.mutation_ids.join('\t')
          samples.times do |saved_index|
            row = Array(String).new(mutation_count)
            mutation_count.times do |mutation_index|
              offset = saved_index * mutation_count * sample_count + mutation_index * sample_count + sample_index
              row << trace[offset].to_s
            end
            file.puts row.join('\t')
          end
        end
      end

      File.open(File.join(dir, "precision.tsv"), "w") do |file|
        result.saved_precision_trace.each do |value|
          file.puts value
        end
      end
    end

    def self.execute(config : Config)
      profile = CliProfile.new
      total_started = Time.instant

      started = Time.instant
      rows = Input.read_tsv(config.in_file)
      profile.input_read_ms = elapsed_ms(started)

      started = Time.instant
      sanitized_rows = Sanitize.run(rows)
      profile.sanitize_ms = elapsed_ms(started)
      if sanitized_rows.empty?
        raise CliError.new("No valid rows remain after sanitization")
      end

      started = Time.instant
      indexed = Indexing.build(sanitized_rows)
      profile.indexing_ms = elapsed_ms(started)

      started = Time.instant
      result = case config.engine
               when Engine::MCMC
                 Kernel.fit_mcmc(config, indexed.rows, indexed.num_mutations, indexed.num_samples)
               else
                 Kernel.fit(config, indexed.rows, indexed.num_mutations, indexed.num_samples)
               end
      profile.kernel_ms = elapsed_ms(started)

      begin
        started = Time.instant
        out_rows = ResultBuilder.build(indexed, result)
        profile.result_build_ms = elapsed_ms(started)

        started = Time.instant
        Output.write(config.out_file, out_rows, config.compress?)
        profile.output_write_ms = elapsed_ms(started)

        if config.engine.mcmc? && (trace_dir = ENV["TOYCLONE_MCMC_TRACE_DIR"]?)
          started = Time.instant
          dump_mcmc_trace(trace_dir, indexed, result)
          profile.trace_dump_ms = elapsed_ms(started)
        end
      ensure
        result.free
        profile.total_ms = elapsed_ms(total_started)
        if ENV["PCV_PROFILE"]?
          profile.print_summary
        end
      end
    end

    private def self.elapsed_ms(started : Time::Instant) : Float64
      (Time.instant - started).total_milliseconds
    end
  end
end
