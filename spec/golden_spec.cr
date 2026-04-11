require "csv"
require "file_utils"
require "uuid"
require "./spec_helper"

private def golden_config(in_file : String, out_file : String)
  config = Tyclone::ViConfig.new
  config.in_file = in_file
  config.out_file = out_file
  config.num_clusters = 4
  config.density = Tyclone::Density::BetaBinomial
  config.num_grid_points = 21
  config.num_restarts = 2
  config.convergence_threshold = 1e-6
  config.max_iters = 200
  config.mix_weight_prior = 1.0
  config.precision = 1000.0
  config.print_freq = 0
  config.seed = 7_u64
  config.kernel_threads = 1
  config.restart_parallelism = 1
  config.compress = false
  config
end

private def load_tsv(path : String)
  rows = [] of NamedTuple(
    mutation_id: String,
    sample_id: String,
    cluster_id: Int32,
    cellular_prevalence: Float64,
    cellular_prevalence_std: Float64,
    cluster_assignment_prob: Float64,
  )

  File.open(path) do |file|
    csv = CSV.new(file, headers: true, separator: '\t')
    csv.each do |row|
      rows << {
        mutation_id:             row["mutation_id"],
        sample_id:               row["sample_id"],
        cluster_id:              row["cluster_id"].to_i,
        cellular_prevalence:     row["cellular_prevalence"].to_f,
        cellular_prevalence_std: row["cellular_prevalence_std"].to_f,
        cluster_assignment_prob: row["cluster_assignment_prob"].to_f,
      }
    end
  end

  rows
end

private def expect_close(actual : Float64, expected : Float64, row_label : String, column_name : String, epsilon : Float64 = 1e-6)
  diff = (actual - expected).abs
  return if diff <= epsilon

  raise "golden mismatch at #{row_label} #{column_name}: expected=#{expected} actual=#{actual} diff=#{diff} epsilon=#{epsilon}"
end

describe "golden output" do
  it "matches deterministic synthetic output within tolerance" do
    in_file = File.expand_path("./fixtures/synthetic_input.tsv", __DIR__)
    expected_file = File.expand_path("./fixtures/synthetic_golden.tsv", __DIR__)
    out_file = File.join(Dir.tempdir, "tyclone-golden-#{UUID.random}.tsv")

    begin
      Tyclone::Run.execute(golden_config(in_file, out_file))

      actual_rows = load_tsv(out_file)
      expected_rows = load_tsv(expected_file)

      actual_rows.size.should eq(expected_rows.size)

      actual_rows.zip(expected_rows).each_with_index do |(actual, expected), index|
        row_label = "line #{index + 2} (mutation_id=#{expected[:mutation_id]}, sample_id=#{expected[:sample_id]}, cluster_id=#{expected[:cluster_id]})"

        actual[:mutation_id].should eq(expected[:mutation_id])
        actual[:sample_id].should eq(expected[:sample_id])
        actual[:cluster_id].should eq(expected[:cluster_id])
        expect_close(actual[:cellular_prevalence], expected[:cellular_prevalence], row_label, "cellular_prevalence")
        expect_close(actual[:cellular_prevalence_std], expected[:cellular_prevalence_std], row_label, "cellular_prevalence_std")
        expect_close(actual[:cluster_assignment_prob], expected[:cluster_assignment_prob], row_label, "cluster_assignment_prob")
      end
    ensure
      File.delete(out_file) if File.exists?(out_file)
    end
  end
end
