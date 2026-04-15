module Tyclone
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
      rows = Input.read_tsv(config.in_file)
      sanitized_rows = Sanitize.run(rows)
      if sanitized_rows.empty?
        raise CliError.new("No valid rows remain after sanitization")
      end

      indexed = Indexing.build(sanitized_rows)
      result = case config.engine
               when Engine::MCMC
                 Kernel.fit_mcmc(config, indexed.rows, indexed.num_mutations, indexed.num_samples)
               else
                 Kernel.fit(config, indexed.rows, indexed.num_mutations, indexed.num_samples)
               end

      begin
        out_rows = ResultBuilder.build(indexed, result)
        Output.write(config.out_file, out_rows, config.compress?)
        if config.engine.mcmc? && (trace_dir = ENV["TOYCLONE_MCMC_TRACE_DIR"]?)
          dump_mcmc_trace(trace_dir, indexed, result)
        end
      ensure
        result.free
      end
    end
  end
end
