require "csv"

module Tyclone
  struct InputRow
    getter mutation_id : String
    getter sample_id : String
    getter ref_counts : Int32
    getter alt_counts : Int32
    getter major_cn : Int32
    getter minor_cn : Int32
    getter normal_cn : Int32
    getter tumour_content : Float64
    getter error_rate : Float64

    def initialize(
      @mutation_id : String,
      @sample_id : String,
      @ref_counts : Int32,
      @alt_counts : Int32,
      @major_cn : Int32,
      @minor_cn : Int32,
      @normal_cn : Int32,
      @tumour_content : Float64,
      @error_rate : Float64,
    )
    end
  end

  module Input
    REQUIRED_COLUMNS = {
      "mutation_id",
      "sample_id",
      "ref_counts",
      "alt_counts",
      "major_cn",
      "minor_cn",
      "normal_cn",
    }

    def self.read_tsv(path : String) : Array(InputRow)
      rows = [] of InputRow
      file = File.open(path)
      csv = CSV.new(file, headers: true, separator: '\t')
      headers = csv.headers || [] of String
      missing = REQUIRED_COLUMNS.reject { |column| headers.includes?(column) }
      unless missing.empty?
        raise CliError.new("Missing required columns: #{missing.join(", ")}")
      end

      csv.each do |row|
        mutation_id = required(row, "mutation_id")
        sample_id = required(row, "sample_id")
        ref_counts = required(row, "ref_counts")
        alt_counts = required(row, "alt_counts")
        major_cn = required(row, "major_cn")
        minor_cn = required(row, "minor_cn")
        normal_cn = required(row, "normal_cn")
        tumour_content = optional(row, "tumour_content", "1.0")
        error_rate = optional(row, "error_rate", "0.001")

        rows << InputRow.new(
          mutation_id: mutation_id,
          sample_id: sample_id,
          ref_counts: ref_counts.to_i,
          alt_counts: alt_counts.to_i,
          major_cn: major_cn.to_i,
          minor_cn: minor_cn.to_i,
          normal_cn: normal_cn.to_i,
          tumour_content: tumour_content.to_f,
          error_rate: error_rate.to_f
        )
      end

      file.close
      rows
    end

    private def self.required(row : CSV, key : String) : String
      value = row[key]?
      if value.nil? || value.empty?
        raise CliError.new("Missing value for '#{key}' in an input row")
      end
      value
    end

    private def self.optional(row : CSV, key : String, default_value : String) : String
      value = row[key]?
      return default_value if value.nil? || value.empty?
      value
    end
  end
end
