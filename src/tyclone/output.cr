require "csv"
require "compress/gzip"

module Tyclone
  module Output
    HEADER = {
      "mutation_id",
      "sample_id",
      "cluster_id",
      "cellular_prevalence",
      "cellular_prevalence_std",
      "cluster_assignment_prob",
    }

    def self.write(path : String, rows : Array(OutputRow), compress : Bool)
      if compress
        File.open(path, "w") do |file|
          Compress::Gzip::Writer.open(file) do |gzip|
            write_io(gzip, rows)
          end
        end
      else
        File.open(path, "w") do |file|
          write_io(file, rows)
        end
      end
    end

    private def self.write_io(io : IO, rows : Array(OutputRow))
      CSV.build(io, separator: '\t') do |csv|
        csv.row(*HEADER)
        rows.each do |row|
          csv.row(
            row.mutation_id,
            row.sample_id,
            row.cluster_id,
            row.cellular_prevalence,
            row.cellular_prevalence_std,
            row.cluster_assignment_prob
          )
        end
      end
    end
  end
end
