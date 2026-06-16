require "csv"

module UnClone
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
      File.open(path) do |file|
        csv = CSV.new(file, headers: true, separator: '\t')
        headers = csv.headers || [] of String
        missing = REQUIRED_COLUMNS.reject { |column| headers.includes?(column) }
        unless missing.empty?
          raise CliError.new("Missing required columns: #{missing.join(", ")}")
        end

        line_number = 1
        csv.each do |row|
          line_number += 1
          mutation_id = required(row, "mutation_id", line_number)
          sample_id = required(row, "sample_id", line_number)
          ref_counts = parse_i32(row, "ref_counts", line_number)
          alt_counts = parse_i32(row, "alt_counts", line_number)
          major_cn = parse_i32(row, "major_cn", line_number)
          minor_cn = parse_i32(row, "minor_cn", line_number)
          normal_cn = parse_i32(row, "normal_cn", line_number)
          tumour_content = parse_f64(row, "tumour_content", line_number, "1.0")
          error_rate = parse_f64(row, "error_rate", line_number, "0.001")

          validate_values(
            line_number,
            ref_counts,
            alt_counts,
            major_cn,
            minor_cn,
            normal_cn,
            tumour_content,
            error_rate
          )

          rows << InputRow.new(
            mutation_id: mutation_id,
            sample_id: sample_id,
            ref_counts: ref_counts,
            alt_counts: alt_counts,
            major_cn: major_cn,
            minor_cn: minor_cn,
            normal_cn: normal_cn,
            tumour_content: tumour_content,
            error_rate: error_rate
          )
        end
      end

      rows
    end

    private def self.required(row : CSV, key : String, line_number : Int32) : String
      value = row[key]?
      if value.nil? || value.empty?
        raise CliError.new("Line #{line_number}: missing value for '#{key}'")
      end
      value
    end

    private def self.optional(row : CSV, key : String, default_value : String) : String
      value = row[key]?
      return default_value if value.nil? || value.empty?
      value
    end

    private def self.parse_i32(row : CSV, key : String, line_number : Int32) : Int32
      value = required(row, key, line_number)
      value.to_i32? || raise CliError.new("Line #{line_number}: invalid integer for '#{key}': #{value}")
    end

    private def self.parse_f64(row : CSV, key : String, line_number : Int32, default_value : String) : Float64
      value = optional(row, key, default_value)
      parsed = value.to_f64?
      unless parsed && parsed.finite?
        raise CliError.new("Line #{line_number}: invalid number for '#{key}': #{value}")
      end
      parsed
    end

    private def self.validate_values(
      line_number : Int32,
      ref_counts : Int32,
      alt_counts : Int32,
      major_cn : Int32,
      minor_cn : Int32,
      normal_cn : Int32,
      tumour_content : Float64,
      error_rate : Float64,
    ) : Nil
      raise CliError.new("Line #{line_number}: ref_counts must be >= 0") if ref_counts < 0
      raise CliError.new("Line #{line_number}: alt_counts must be >= 0") if alt_counts < 0
      raise CliError.new("Line #{line_number}: major_cn must be >= 0") if major_cn < 0
      raise CliError.new("Line #{line_number}: minor_cn must be >= 0") if minor_cn < 0
      raise CliError.new("Line #{line_number}: normal_cn must be >= 0") if normal_cn < 0
      raise CliError.new("Line #{line_number}: minor_cn must be <= major_cn") if minor_cn > major_cn
      raise CliError.new("Line #{line_number}: tumour_content must be within [0, 1]") unless (0.0..1.0).includes?(tumour_content)
      raise CliError.new("Line #{line_number}: error_rate must be within [0, 1]") unless (0.0..1.0).includes?(error_rate)
    end
  end
end
