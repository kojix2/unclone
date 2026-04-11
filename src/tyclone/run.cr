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
    private def self.load_indexed_rows(in_file : String, profile : CliProfile) : IndexedRows
      started = Time.instant
      rows = Input.read_tsv(in_file)
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
      indexed
    end

    private def self.write_result(out_file : String, compress : Bool, indexed : IndexedRows, result : TabularKernelResult, profile : CliProfile)
      started = Time.instant
      out_rows = ResultBuilder.build(indexed, result)
      profile.result_build_ms = elapsed_ms(started)

      started = Time.instant
      Output.write(out_file, out_rows, compress)
      profile.output_write_ms = elapsed_ms(started)
    end

    def self.execute(config : ViConfig)
      profile = CliProfile.new
      total_started = Time.instant

      indexed = load_indexed_rows(config.in_file, profile)

      started = Time.instant
      result = Kernel.fit(config, indexed.rows, indexed.num_mutations, indexed.num_samples)
      profile.kernel_ms = elapsed_ms(started)

      begin
        write_result(config.out_file, config.compress?, indexed, result, profile)
      ensure
        result.free
        profile.total_ms = elapsed_ms(total_started)
        if ENV["PCV_PROFILE"]?
          profile.print_summary
        end
      end
    end

    def self.execute(config : PhyCloneRunConfig)
      PhyClone.run(config)
    end

    def self.execute(config : PhyCloneMapConfig)
      PhyClone.map(config)
    end

    def self.execute(config : PhyCloneConsensusConfig)
      PhyClone.consensus(config)
    end

    def self.execute(config : PhyCloneTopologyReportConfig)
      PhyClone.topology_report(config)
    end

    private def self.elapsed_ms(started : Time::Instant) : Float64
      (Time.instant - started).total_milliseconds
    end
  end
end
