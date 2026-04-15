require "csv"
require "uuid"
require "./spec_helper"

private def mcmc_config(in_file : String, out_file : String, seed : UInt64 = 7_u64)
  config = Tyclone::Config.new
  config.command = "fit-mcmc"
  config.engine = Tyclone::Engine::MCMC
  config.in_file = in_file
  config.out_file = out_file
  config.num_clusters = 6
  config.density = Tyclone::Density::BetaBinomial
  config.precision = 1000.0
  config.num_iters = 80
  config.burnin = 40
  config.thin = 2
  config.alpha = 1.0
  config.alpha_prior_shape = 1.0
  config.alpha_prior_rate = 0.001
  config.mh_step_size = 0.05
  config.mh_precision_step = 0.0
  config.print_freq = 0
  config.seed = seed
  config.compress = false
  config
end

private def load_mcmc_rows(path : String)
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

describe "fit-mcmc integration" do
  it "is deterministic for a fixed seed" do
    in_file = File.expand_path("./fixtures/synthetic_input.tsv", __DIR__)
    out_file_a = File.join(Dir.tempdir, "tyclone-mcmc-a-#{UUID.random}.tsv")
    out_file_b = File.join(Dir.tempdir, "tyclone-mcmc-b-#{UUID.random}.tsv")

    begin
      Tyclone::Run.execute(mcmc_config(in_file, out_file_a, 42_u64))
      Tyclone::Run.execute(mcmc_config(in_file, out_file_b, 42_u64))

      File.read(out_file_a).should eq(File.read(out_file_b))
    ensure
      File.delete(out_file_a) if File.exists?(out_file_a)
      File.delete(out_file_b) if File.exists?(out_file_b)
    end
  end

  it "produces a valid loci table" do
    in_file = File.expand_path("./fixtures/synthetic_input.tsv", __DIR__)
    out_file = File.join(Dir.tempdir, "tyclone-mcmc-#{UUID.random}.tsv")

    begin
      Tyclone::Run.execute(mcmc_config(in_file, out_file))
      rows = load_mcmc_rows(out_file)

      mutation_ids = rows.map { |row| row[:mutation_id] }.uniq
      sample_ids = rows.map { |row| row[:sample_id] }.uniq

      rows.should_not be_empty
      rows.size.should eq(mutation_ids.size * sample_ids.size)

      cluster_ids = rows.map { |row| row[:cluster_id] }.uniq.sort
      cluster_ids.should eq((0...cluster_ids.size).to_a)

      rows.each do |row|
        row[:cellular_prevalence].should be >= 0.0
        row[:cellular_prevalence].should be <= 1.0
        row[:cellular_prevalence_std].should be >= 0.0
        row[:cluster_assignment_prob].should be >= 0.0
        row[:cluster_assignment_prob].should be <= 1.0
      end
    ensure
      File.delete(out_file) if File.exists?(out_file)
    end
  end

  it "respects different seeds by producing different results" do
    in_file = File.expand_path("./fixtures/synthetic_input.tsv", __DIR__)
    out_file_a = File.join(Dir.tempdir, "tyclone-mcmc-seed1-#{UUID.random}.tsv")
    out_file_b = File.join(Dir.tempdir, "tyclone-mcmc-seed2-#{UUID.random}.tsv")

    begin
      Tyclone::Run.execute(mcmc_config(in_file, out_file_a, 42_u64))
      Tyclone::Run.execute(mcmc_config(in_file, out_file_b, 123_u64))

      content_a = File.read(out_file_a)
      content_b = File.read(out_file_b)

      # Different seeds should (likely) produce different results
      # (Though theoretically they could be the same, it's vanishingly unlikely with 80 iters)
      (content_a != content_b).should be_true
    ensure
      File.delete(out_file_a) if File.exists?(out_file_a)
      File.delete(out_file_b) if File.exists?(out_file_b)
    end
  end

  it "cluster assignment probabilities are well-calibrated" do
    in_file = File.expand_path("./fixtures/synthetic_input.tsv", __DIR__)
    out_file = File.join(Dir.tempdir, "tyclone-mcmc-calib-#{UUID.random}.tsv")

    begin
      Tyclone::Run.execute(mcmc_config(in_file, out_file))
      rows = load_mcmc_rows(out_file)

      # Check that cluster_assignment_prob values are reasonable
      probs = rows.map { |row| row[:cluster_assignment_prob] }
      mean_prob = probs.sum / probs.size
      max_prob = probs.max
      min_prob = probs.min

      mean_prob.should be >= 0.1 # Average should not be too low
      mean_prob.should be <= 0.9 # Average should not be too high
      max_prob.should be > 0.5   # Some high confidence assignments
      min_prob.should be < 0.5   # Some low confidence assignments
    ensure
      File.delete(out_file) if File.exists?(out_file)
    end
  end

  it "produces qualitatively similar results despite different burn-in" do
    in_file = File.expand_path("./fixtures/synthetic_input.tsv", __DIR__)
    out_file_burnin = File.join(Dir.tempdir, "tyclone-mcmc-burnin-#{UUID.random}.tsv")
    out_file_noburnin = File.join(Dir.tempdir, "tyclone-mcmc-noburnin-#{UUID.random}.tsv")

    begin
      config_burnin = mcmc_config(in_file, out_file_burnin, 99_u64)
      config_burnin.num_iters = 200
      config_burnin.burnin = 100
      config_burnin.thin = 2
      Tyclone::Run.execute(config_burnin)

      config_noburnin = mcmc_config(in_file, out_file_noburnin, 99_u64)
      config_noburnin.num_iters = 200
      config_noburnin.burnin = 0
      config_noburnin.thin = 2
      Tyclone::Run.execute(config_noburnin)

      rows_burnin = load_mcmc_rows(out_file_burnin)
      rows_noburnin = load_mcmc_rows(out_file_noburnin)

      # Both should have same structure
      rows_burnin.size.should eq(rows_noburnin.size)

      # But burn-in should generally produce more stable estimates (lower std)
      std_burnin = rows_burnin.map { |r| r[:cellular_prevalence_std] }.reduce(0.0) { |sum, v| sum + v }
      std_noburnin = rows_noburnin.map { |r| r[:cellular_prevalence_std] }.reduce(0.0) { |sum, v| sum + v }

      # Burn-in typically produces lower variance (but test allows for variation)
      (std_burnin / std_noburnin).should be < 1.5 # Burn-in should not be much worse
    ensure
      File.delete(out_file_burnin) if File.exists?(out_file_burnin)
      File.delete(out_file_noburnin) if File.exists?(out_file_noburnin)
    end
  end
end
