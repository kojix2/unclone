require "csv"
require "file_utils"
require "uuid"
require "./spec_helper"

private def missing_observation_config(in_file : String, out_file : String)
  config = UnClone::ViConfig.new
  config.in_file = in_file
  config.out_file = out_file
  config.num_clusters = 5
  config.density = UnClone::Density::BetaBinomial
  config.num_grid_points = 11
  config.num_restarts = 1
  config.convergence_threshold = 1e-4
  config.max_iters = 30
  config.mix_weight_prior = 1.0
  config.precision = 200.0
  config.print_freq = 0
  config.seed = 1_u64
  config.kernel_threads = 1
  config.restart_parallelism = 1
  config.compress = false
  config
end

private def run_missing_observation_case(input : String)
  dir = File.join(Dir.tempdir, "unclone-missing-observation-#{UUID.random}")
  FileUtils.mkdir_p(dir)
  in_file = File.join(dir, "input.tsv")
  out_file = File.join(dir, "out.tsv")

  begin
    File.write(in_file, input)
    UnClone::Run.execute(missing_observation_config(in_file, out_file))

    rows = [] of NamedTuple(mutation_id: String, sample_id: String)
    File.open(out_file) do |file|
      csv = CSV.new(file, headers: true, separator: '\t')
      csv.each do |row|
        rows << {mutation_id: row["mutation_id"], sample_id: row["sample_id"]}
      end
    end
    rows
  ensure
    FileUtils.rm_rf(dir) if Dir.exists?(dir)
  end
end

private def expect_mut_a_and_mut_b_for_all_samples(rows)
  mutation_ids = rows.map(&.[:mutation_id])
  mutation_ids.uniq!
  mutation_ids.sort!
  mutation_ids.should eq(["mutA", "mutB"])

  rows.size.should eq(4)
  rows.count { |row| row[:mutation_id] == "mutA" }.should eq(2)
  rows.count { |row| row[:mutation_id] == "mutB" }.should eq(2)
  rows.map(&.[:sample_id]).uniq!.sort!.should eq(["S1", "S2"])
end

describe "VI missing observations" do
  it "accepts zero-depth rows and keeps the mutation" do
    rows = run_missing_observation_case(
      "mutation_id\tsample_id\tref_counts\talt_counts\tnormal_cn\tminor_cn\tmajor_cn\n" +
      "mutA\tS1\t50\t50\t2\t1\t1\n" +
      "mutA\tS2\t0\t0\t2\t1\t1\n" +
      "mutB\tS1\t80\t20\t2\t1\t1\n" +
      "mutB\tS2\t40\t60\t2\t1\t1\n"
    )

    expect_mut_a_and_mut_b_for_all_samples(rows)
  end

  it "keeps mutations observed in only one sample" do
    rows = run_missing_observation_case(
      "mutation_id\tsample_id\tref_counts\talt_counts\tnormal_cn\tminor_cn\tmajor_cn\n" +
      "mutA\tS1\t50\t50\t2\t1\t1\n" +
      "mutB\tS1\t80\t20\t2\t1\t1\n" +
      "mutB\tS2\t40\t60\t2\t1\t1\n"
    )

    expect_mut_a_and_mut_b_for_all_samples(rows)
  end
end
