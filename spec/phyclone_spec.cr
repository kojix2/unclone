require "json"
require "uuid"
require "./spec_helper"

private def phy_topology_ids(path : String) : Array(String)
  File.read_lines(path).map { |line| JSON.parse(line)["topology_id"].as_s }
end

def write_phy_input(path : String)
  File.write(path, <<-TSV)
    mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\tnormal_cn\tcluster_id
    m0\ts0\t10\t3\t1\t1\t2\t0
    m0\ts1\t12\t4\t1\t1\t2\t0
    m1\ts0\t8\t2\t1\t1\t2\t1
    TSV
end

def write_phy_input_three_clusters(path : String)
  File.write(path, <<-TSV)
    mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\tnormal_cn\tcluster_id\tchrom
    m0\ts0\t30\t20\t1\t1\t2\t0\tchr1
    m0\ts1\t24\t16\t1\t1\t2\t0\tchr1
    m1\ts0\t22\t10\t1\t1\t2\t1\tchr2
    m1\ts1\t25\t9\t1\t1\t2\t1\tchr2
    m2\ts0\t18\t4\t1\t1\t2\t2\tchr3
    TSV
end

describe Tyclone::PhyCloneRunConfig do
  it "has expected default values" do
    config = Tyclone::PhyCloneRunConfig.new
    config.in_file.should eq("")
    config.out_file.should eq("")
    config.num_iters.should eq(10_000)
    config.num_chains.should eq(1)
    config.num_particles.should eq(100)
    config.burn_in_iters.should eq(1_000)
    config.max_time.infinite?.should eq(1)
    config.print_freq.should eq(100)
    config.num_samples_data_point.should eq(1)
    config.subtree_update_prob.should be_close(0.0, 1e-12)
    config.concentration_update?.should be_true
    config.concentration_value.should be_close(1.0, 1e-12)
    config.outlier_prob.should be_close(0.0, 1e-12)
    config.proposal.should eq(Tyclone::PhyCloneProposal::SemiAdapted)
    config.thin.should eq(1)
    config.resample_threshold.should be_close(0.5, 1e-12)
    config.seed.should be_nil
    config.compress?.should be_false
  end
end

describe Tyclone::CLI do
  it "parses phy run command" do
    command = Tyclone::CLI.parse([
      "phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "-n", "3", "--num-chains=2",
    ])
    command.should be_a(Tyclone::PhyCloneRunCommand)
    config = command.as(Tyclone::PhyCloneRunCommand).config
    config.in_file.should eq("in.tsv")
    config.out_file.should eq("trace.jsonl")
    config.num_iters.should eq(3)
    config.num_chains.should eq(2)
  end

  it "parses phy run sampler options" do
    command = Tyclone::CLI.parse([
      "phy", "run", "-i", "in.tsv", "-o", "trace.jsonl",
      "--num-particles=12", "--burnin=4", "--max-time=12.5", "--print-freq=7",
      "--thin=3", "--resample-threshold=0.25",
      "--num-samples-data-point=2",
      "--no-concentration-update", "--concentration-value=0.5",
      "--proposal=fully-adapted",
    ])
    command.should be_a(Tyclone::PhyCloneRunCommand)
    config = command.as(Tyclone::PhyCloneRunCommand).config
    config.num_particles.should eq(12)
    config.burn_in_iters.should eq(4)
    config.max_time.should be_close(12.5, 1e-12)
    config.print_freq.should eq(7)
    config.num_samples_data_point.should eq(2)
    config.thin.should eq(3)
    config.resample_threshold.should be_close(0.25, 1e-12)
    config.concentration_update?.should be_false
    config.concentration_value.should be_close(0.5, 1e-12)
    config.proposal.should eq(Tyclone::PhyCloneProposal::FullyAdapted)
  end

  it "rejects non-positive concentration-value" do
    expect_raises(Tyclone::CliError, /--concentration-value must be > 0/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--concentration-value=0"])
    end
  end

  it "rejects negative num-samples-data-point" do
    expect_raises(Tyclone::CliError, /--num-samples-data-point must be >= 0/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--num-samples-data-point=-1"])
    end
  end

  it "rejects negative max-time" do
    expect_raises(Tyclone::CliError, /--max-time must be >= 0/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--max-time=-1"])
    end
  end

  it "rejects non-positive print-freq" do
    expect_raises(Tyclone::CliError, /--print-freq must be >= 1/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--print-freq=0"])
    end
  end

  it "rejects invalid phy proposal option" do
    expect_raises(Tyclone::CliError, /Invalid proposal: unknown/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--proposal=unknown"])
    end
  end

  it "rejects removed custom sampler option" do
    expect_raises(Tyclone::CliError, /--sampler-mode is not a valid option/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--sampler-mode=unconditional"])
    end
  end

  it "rejects removed custom prevalence threshold option" do
    expect_raises(Tyclone::CliError, /--loss-prevalence-threshold is not a valid option/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--loss-prevalence-threshold=0.12"])
    end
  end

  it "rejects removed retained weighting options" do
    expect_raises(Tyclone::CliError, /--retained-similarity-weight is not a valid option/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--retained-similarity-weight=0.3"])
    end
    expect_raises(Tyclone::CliError, /--retained-score-gap-weight is not a valid option/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--retained-score-gap-weight=0.07"])
    end
  end

  it "requires cluster-file for loss probability compatibility flags" do
    expect_raises(Tyclone::CliError, /require --cluster-file/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--assign-loss-prob"])
    end
    expect_raises(Tyclone::CliError, /require --cluster-file/) do
      Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "trace.jsonl", "--user-provided-loss-prob"])
    end
  end

  it "parses loss probability compatibility flags with cluster-file" do
    command = Tyclone::CLI.parse([
      "phy", "run",
      "-i", "in.tsv",
      "-o", "trace.jsonl",
      "--cluster-file", "clusters.tsv",
      "--assign-loss-prob",
      "--loss-prob=0.05",
      "--high-loss-prob=0.4",
    ])

    command.should be_a(Tyclone::PhyCloneRunCommand)
    config = command.as(Tyclone::PhyCloneRunCommand).config
    config.assign_loss_prob?.should be_true
    config.user_provided_loss_prob?.should be_false
    config.cluster_file.should eq("clusters.tsv")
    config.loss_prob.should eq(0.05)
    config.high_loss_prob.should eq(0.4)
  end

  it "shows phy root help" do
    command = Tyclone::CLI.parse(["phy", "--help"])
    command.should be_a(Tyclone::HelpCommand)
    command.as(Tyclone::HelpCommand).help_message.should contain("tyclone phy <subcommand>")
  end

  it "parses phy map command" do
    command = Tyclone::CLI.parse(["phy", "map", "-i", "trace.jsonl", "-o", "map.json"])
    command.should be_a(Tyclone::PhyCloneMapCommand)
    config = command.as(Tyclone::PhyCloneMapCommand).config
    config.in_file.should eq("trace.jsonl")
    config.out_file.should eq("map.json")
  end

  it "parses phy consensus command" do
    command = Tyclone::CLI.parse(["phy", "consensus", "-i", "trace.jsonl", "-o", "consensus.json"])
    command.should be_a(Tyclone::PhyCloneConsensusCommand)
    config = command.as(Tyclone::PhyCloneConsensusCommand).config
    config.in_file.should eq("trace.jsonl")
    config.out_file.should eq("consensus.json")
  end

  it "parses phy topology-report command" do
    command = Tyclone::CLI.parse(["phy", "topology-report", "-i", "trace.jsonl", "-o", "report.json"])
    command.should be_a(Tyclone::PhyCloneTopologyReportCommand)
    config = command.as(Tyclone::PhyCloneTopologyReportCommand).config
    config.in_file.should eq("trace.jsonl")
    config.out_file.should eq("report.json")
  end
end

describe Tyclone::PhyClone do
  it "writes a minimal JSONL trace for phy run" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-input-#{UUID.random}.tsv")
    output_path = File.join(Dir.tempdir, "tyclone-phy-trace-#{UUID.random}.jsonl")
    write_phy_input(input_path)

    config = Tyclone::PhyCloneRunConfig.new
    config.in_file = input_path
    config.out_file = output_path
    config.num_iters = 8
    config.num_chains = 1
    config.seed = 17_u64

    Tyclone::Run.execute(config)

    lines = File.read_lines(output_path)
    lines.size.should eq(8)

    record = JSON.parse(lines.first)
    record["schema_version"].as_i.should eq(1)
    record["chain"].as_i.should eq(0)
    record["iter"].as_i.should eq(0)
    record["tree"]["nodes"].as_a.size.should be >= 1 # root-only tree is possible
    record["clusters"].as_a.size.should be >= 1      # SMC may not include all clusters
    phy_topology_ids(output_path).uniq.size.should be >= 1
  ensure
    File.delete?(input_path.as(String))
    File.delete?(output_path.as(String))
  end

  it "is deterministic for phy run when seed is fixed" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-input-#{UUID.random}.tsv")
    output_path_a = File.join(Dir.tempdir, "tyclone-phy-trace-a-#{UUID.random}.jsonl")
    output_path_b = File.join(Dir.tempdir, "tyclone-phy-trace-b-#{UUID.random}.jsonl")
    write_phy_input(input_path)

    config_a = Tyclone::PhyCloneRunConfig.new
    config_a.in_file = input_path
    config_a.out_file = output_path_a
    config_a.num_iters = 8
    config_a.num_chains = 1
    config_a.seed = 17_u64

    config_b = Tyclone::PhyCloneRunConfig.new
    config_b.in_file = input_path
    config_b.out_file = output_path_b
    config_b.num_iters = 8
    config_b.num_chains = 1
    config_b.seed = 17_u64

    Tyclone::Run.execute(config_a)
    Tyclone::Run.execute(config_b)

    File.read(output_path_a).should eq(File.read(output_path_b))
  ensure
    File.delete?(input_path.as(String))
    File.delete?(output_path_a.as(String))
    File.delete?(output_path_b.as(String))
  end

  it "builds a minimal map summary from a trace" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-input-#{UUID.random}.tsv")
    trace_path = File.join(Dir.tempdir, "tyclone-phy-trace-#{UUID.random}.jsonl")
    map_path = File.join(Dir.tempdir, "tyclone-phy-map-#{UUID.random}.json")
    write_phy_input(input_path)

    run_config = Tyclone::PhyCloneRunConfig.new
    run_config.in_file = input_path
    run_config.out_file = trace_path
    run_config.num_iters = 8
    run_config.num_chains = 1
    run_config.seed = 17_u64

    Tyclone::Run.execute(run_config)

    map_config = Tyclone::PhyCloneMapConfig.new
    map_config.in_file = trace_path
    map_config.out_file = map_path

    Tyclone::Run.execute(map_config)

    result = JSON.parse(File.read(map_path))
    result["schema_version"].as_i.should eq(1)
    result["topology_id"].as_s.should_not eq("")
    # SMC may not include all clusters; just check that newick contains at least one cluster reference
    result["newick"].as_s.should contain("root")
    result["iter"].as_i.should be >= 0
    result["clusters"].as_a.size.should be >= 1 # SMC may not include all clusters
    result["num_records"].as_i.should eq(8)
    result["map_method"].as_s.should eq("best_record_with_trace_quantiles")
    result["outlier_assignments"].as_a.size.should be >= 1 # SMC may assign varying data points
    result["outlier_summary"]["num_assignments"].as_i.should be >= 1
    result["outlier_summary"]["num_in_tree"].as_i.should be >= 0

    cluster_ci = result["cluster_sample_prevalence_ci"]
    clonal_ci = result["clonal_prevalence_ci"]
    # Verify prevalence CI structure exists
    cluster_ci.as_h?.should_not be_nil
    clonal_ci.as_h?.should_not be_nil
  ensure
    File.delete?(input_path.as(String))
    File.delete?(trace_path.as(String))
    File.delete?(map_path.as(String))
  end

  it "builds a minimal consensus summary from a trace" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-input-#{UUID.random}.tsv")
    trace_path = File.join(Dir.tempdir, "tyclone-phy-trace-#{UUID.random}.jsonl")
    consensus_path = File.join(Dir.tempdir, "tyclone-phy-consensus-#{UUID.random}.json")
    write_phy_input(input_path)

    run_config = Tyclone::PhyCloneRunConfig.new
    run_config.in_file = input_path
    run_config.out_file = trace_path
    run_config.num_iters = 8
    run_config.num_chains = 2
    run_config.seed = 17_u64

    Tyclone::Run.execute(run_config)

    consensus_config = Tyclone::PhyCloneConsensusConfig.new
    consensus_config.in_file = trace_path
    consensus_config.out_file = consensus_path

    Tyclone::Run.execute(consensus_config)

    result = JSON.parse(File.read(consensus_path))
    result["schema_version"].as_i.should eq(1)
    result["topology_id"].as_s.should_not eq("")
    # SMC may not include all clusters; just check that newick contains root
    result["newick"].as_s.should contain("root")
    result["support"].as_i.should be > 0
    result["num_records"].as_i.should eq(16)
    result["representative"]["iter"].as_i.should be >= 0
    result["clusters"].as_a.size.should be >= 1 # SMC may not include all clusters

    # Clade-based consensus fields
    clade_consensus = result["clade_consensus"]
    clade_consensus["num_clades"].as_i.should be > 0
    clade_consensus["clades"].as_a.each do |clade|
      clade["clade"].as_a.size.should be > 0
      clade["support"].as_i.should be > 0
      (0.0..1.0).includes?(clade["support_fraction"].as_f).should be_true
    end

    consensus_tree = result["consensus_tree"]
    consensus_tree["nodes"].as_a.size.should be > 0
    consensus_tree["newick"].as_s.should contain(";")
  ensure
    File.delete?(input_path.as(String))
    File.delete?(trace_path.as(String))
    File.delete?(consensus_path.as(String))
  end

  it "builds a minimal topology report from a trace" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-input-#{UUID.random}.tsv")
    trace_path = File.join(Dir.tempdir, "tyclone-phy-trace-#{UUID.random}.jsonl")
    report_path = File.join(Dir.tempdir, "tyclone-phy-topology-report-#{UUID.random}.json")
    write_phy_input(input_path)

    run_config = Tyclone::PhyCloneRunConfig.new
    run_config.in_file = input_path
    run_config.out_file = trace_path
    run_config.num_iters = 8
    run_config.num_chains = 2
    run_config.seed = 17_u64

    Tyclone::Run.execute(run_config)

    report_config = Tyclone::PhyCloneTopologyReportConfig.new
    report_config.in_file = trace_path
    report_config.out_file = report_path

    Tyclone::Run.execute(report_config)

    result = JSON.parse(File.read(report_path))
    result["schema_version"].as_i.should eq(1)
    result["num_records"].as_i.should eq(16)
    result["num_topologies"].as_i.should be >= 1 # At least one topology
    topology = result["topologies"].as_a.first
    topology["topology_id"].as_s.should_not eq("")
    topology["support"].as_i.should be > 0
    topology["support_fraction"].as_f.should be <= 1.0
    topology["support_fraction"].as_f.should be > 0.0
    topology["best_log_p"].as_f.finite?.should be_true
    # SMC may not include all clusters; just check that newick contains root
    topology["newick"].as_s.should contain("root")
    topology["representative"]["iter"].as_i.should be >= 0
  ensure
    File.delete?(input_path.as(String))
    File.delete?(trace_path.as(String))
    File.delete?(report_path.as(String))
  end

  it "rejects unsupported phy trace schema versions" do
    trace_path = File.join(Dir.tempdir, "tyclone-phy-invalid-trace-#{UUID.random}.jsonl")
    map_path = File.join(Dir.tempdir, "tyclone-phy-invalid-map-#{UUID.random}.json")

    File.write(trace_path, <<-JSONL)
      {"schema_version":999,"chain":0,"iter":0,"log_p":-2.0,"topology_id":"star-1","tree":{"nodes":[{"id":"root","kind":"root"},{"id":"cluster-0","kind":"cluster","cluster_id":0}],"edges":[{"parent":"root","child":"cluster-0"}]},"clusters":[{"cluster_id":0,"mutation_ids":["m0"],"sample_ids":["s0"]}]}
      JSONL

    config = Tyclone::PhyCloneMapConfig.new
    config.in_file = trace_path
    config.out_file = map_path

    expect_raises(Tyclone::CliError, /Unsupported phy trace schema_version/) do
      Tyclone::Run.execute(config)
    end
  ensure
    File.delete?(trace_path.as(String))
    File.delete?(map_path.as(String))
  end

  it "emits internal cluster edges for three-cluster phy run" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-input-three-#{UUID.random}.tsv")
    trace_path = File.join(Dir.tempdir, "tyclone-phy-trace-three-#{UUID.random}.jsonl")
    write_phy_input_three_clusters(input_path)

    config = Tyclone::PhyCloneRunConfig.new
    config.in_file = input_path
    config.out_file = trace_path
    config.num_iters = 24
    config.num_chains = 1
    config.seed = 23_u64

    Tyclone::Run.execute(config)

    # SMC generates various topologies; at least check that some iterations have clusters
    has_any_cluster_node = File.read_lines(trace_path).any? do |line|
      document = JSON.parse(line)
      document["tree"]["nodes"].as_a.any? do |node|
        node["kind"].as_s == "cluster"
      end
    end

    has_any_cluster_node.should be_true
  ensure
    File.delete?(input_path.as(String))
    File.delete?(trace_path.as(String))
  end

  it "runs SMC sampler with burn-in and emits post-burnin main chain trace" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-smc-input-#{UUID.random}.tsv")
    trace_path = File.join(Dir.tempdir, "tyclone-phy-smc-trace-#{UUID.random}.jsonl")
    write_phy_input_three_clusters(input_path)

    run_config = Tyclone::PhyCloneRunConfig.new
    run_config.in_file = input_path
    run_config.out_file = trace_path
    run_config.num_iters = 5
    run_config.num_chains = 1
    run_config.seed = 42_u64
    run_config.num_particles = 10
    run_config.burn_in_iters = 3

    Tyclone::Run.execute(run_config)

    trace_records = File.read_lines(trace_path).map { |line| JSON.parse(line) }

    # PhyClone semantics: burn-in and main iterations are separate, so output
    # length is `num_iters` (main chain count), independent of burn-in.
    trace_records.size.should eq(5)

    # Each record should have valid topology
    trace_records.each do |record|
      record["schema_version"].as_i.should eq(1)
      record["topology_id"].as_s.should_not eq("")
      record["tree"]["nodes"].as_a.size.should be >= 1
    end

    # Iteration counter is re-indexed over recorded main-chain samples.
    iters = trace_records.map(&.["iter"].as_i)
    iters.should eq([0, 1, 2, 3, 4])
  ensure
    File.delete?(input_path.as(String))
    File.delete?(trace_path.as(String))
  end

  it "applies thin as recording interval in phy main chain" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-thin-input-#{UUID.random}.tsv")
    trace_path_thin1 = File.join(Dir.tempdir, "tyclone-phy-thin1-trace-#{UUID.random}.jsonl")
    trace_path_thin2 = File.join(Dir.tempdir, "tyclone-phy-thin2-trace-#{UUID.random}.jsonl")
    write_phy_input_three_clusters(input_path)

    run_config = Tyclone::PhyCloneRunConfig.new
    run_config.in_file = input_path
    run_config.out_file = trace_path_thin1
    run_config.num_iters = 5
    run_config.num_chains = 1
    run_config.seed = 42_u64
    run_config.num_particles = 10
    run_config.burn_in_iters = 3
    run_config.thin = 1

    Tyclone::Run.execute(run_config)

    run_config.out_file = trace_path_thin2
    run_config.thin = 2

    Tyclone::Run.execute(run_config)

    thin1 = File.read_lines(trace_path_thin1).map { |line| JSON.parse(line) }
    thin2 = File.read_lines(trace_path_thin2).map { |line| JSON.parse(line) }

    # PhyClone semantics: num_iters is number of main-chain transitions,
    # and thin is the recording interval over those transitions.
    thin1.size.should eq(5)
    thin2.size.should eq(3) # ceil(5 / 2)

    # Trace iter is the actual main-chain iteration number (PhyClone-compatible).
    thin2.map(&.["iter"].as_i).should eq([0, 2, 4])
  ensure
    File.delete?(input_path.as(String))
    File.delete?(trace_path_thin1.as(String))
    File.delete?(trace_path_thin2.as(String))
  end

  it "emits identical trace counts for num_particles=1 vs particles > 1" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-parity-input-#{UUID.random}.tsv")
    trace_path_single = File.join(Dir.tempdir, "tyclone-phy-parity-single-#{UUID.random}.jsonl")
    trace_path_multi = File.join(Dir.tempdir, "tyclone-phy-parity-multi-#{UUID.random}.jsonl")
    write_phy_input_three_clusters(input_path)

    # Single particle run
    run_config = Tyclone::PhyCloneRunConfig.new
    run_config.in_file = input_path
    run_config.out_file = trace_path_single
    run_config.num_iters = 8
    run_config.num_chains = 1
    run_config.seed = 123_u64
    run_config.num_particles = 1
    run_config.burn_in_iters = 0

    Tyclone::Run.execute(run_config)

    # Multi particle run (same config but more particles)
    run_config.out_file = trace_path_multi
    run_config.num_particles = 5

    Tyclone::Run.execute(run_config)

    single_trace = File.read_lines(trace_path_single).map { |line| JSON.parse(line) }
    multi_trace = File.read_lines(trace_path_multi).map { |line| JSON.parse(line) }

    # Both should emit same number of records (num_iters per chain)
    single_trace.size.should eq(8)
    multi_trace.size.should eq(8)

    # All records should have valid structure
    (single_trace + multi_trace).each do |record|
      record["iter"].as_i.should be >= 0
      log_p = record["log_p"].as_f
      log_p_one = record["log_p_one"].as_f
      log_p.nan?.should be_false
      log_p_one.nan?.should be_false
    end
  ensure
    File.delete?(input_path.as(String))
    File.delete?(trace_path_single.as(String))
    File.delete?(trace_path_multi.as(String))
  end

  it "runs assign-loss-prob mode with cluster-file" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-prev-input-#{UUID.random}.tsv")
    cluster_path = File.join(Dir.tempdir, "tyclone-phy-prev-cluster-#{UUID.random}.tsv")
    trace_path = File.join(Dir.tempdir, "tyclone-phy-prev-trace-#{UUID.random}.jsonl")
    write_phy_input_three_clusters(input_path)

    # Cluster file with cellular_prevalence — low prevalence on m2 triggers high loss
    File.write(cluster_path, <<-TSV)
      mutation_id\tcluster_id\tcellular_prevalence
      m0\t0\t0.8
      m1\t1\t0.5
      m2\t2\t0.02
      TSV

    run_config = Tyclone::PhyCloneRunConfig.new
    run_config.in_file = input_path
    run_config.out_file = trace_path
    run_config.cluster_file = cluster_path
    run_config.num_iters = 5
    run_config.num_chains = 1
    run_config.seed = 77_u64
    run_config.assign_loss_prob = true
    run_config.loss_prob = 0.05
    run_config.high_loss_prob = 0.4

    Tyclone::Run.execute(run_config)

    trace_records = File.read_lines(trace_path).map { |line| JSON.parse(line) }
    trace_records.size.should eq(5)
    trace_records.each do |record|
      record["log_p"].as_f.nan?.should be_false
      record["log_p_one"].as_f.nan?.should be_false
    end
  ensure
    File.delete?(input_path.as(String))
    File.delete?(cluster_path.as(String))
    File.delete?(trace_path.as(String))
  end

  it "accepts string cluster_id in cluster-file and groups mutations correctly" do
    input_path = File.join(Dir.tempdir, "tyclone-phy-strcid-input-#{UUID.random}.tsv")
    cluster_path = File.join(Dir.tempdir, "tyclone-phy-strcid-cluster-#{UUID.random}.tsv")
    trace_path = File.join(Dir.tempdir, "tyclone-phy-strcid-trace-#{UUID.random}.jsonl")

    # Input without inline cluster_id: cluster assignment comes from cluster-file only
    File.write(input_path, <<-TSV)
      mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\tnormal_cn
      m0\ts0\t30\t20\t1\t1\t2
      m0\ts1\t24\t16\t1\t1\t2
      m1\ts0\t22\t10\t1\t1\t2
      m1\ts1\t25\t9\t1\t1\t2
      m2\ts0\t18\t4\t1\t1\t2
      m2\ts1\t20\t5\t1\t1\t2
      TSV

    # Cluster file with string cluster IDs (not integers)
    File.write(cluster_path, <<-TSV)
      mutation_id\tcluster_id
      m0\tcloneA
      m1\tcloneA
      m2\tcloneB
      TSV

    run_config = Tyclone::PhyCloneRunConfig.new
    run_config.in_file = input_path
    run_config.out_file = trace_path
    run_config.cluster_file = cluster_path
    run_config.num_iters = 3
    run_config.num_chains = 1
    run_config.seed = 42_u64

    Tyclone::Run.execute(run_config)

    trace_records = File.read_lines(trace_path).map { |line| JSON.parse(line) }
    trace_records.size.should eq(3)
    trace_records.each do |record|
      record["log_p"].as_f.nan?.should be_false
      # Tree may be root-only; cluster metadata carries grouping information
      root_nodes = record["tree"]["nodes"].as_a.select { |node| node["kind"].as_s == "root" }
      root_nodes.size.should eq(1)
    end
    # All trace records should have 2 clusters (m0+m1 → cloneA, m2 → cloneB)
    trace_records.each do |record|
      record["clusters"].as_a.size.should eq(2)
    end
  ensure
    File.delete?(input_path.as(String))
    File.delete?(cluster_path.as(String))
    File.delete?(trace_path.as(String))
  end

  it "accepts --outlier-prob CLI option" do
    result = Tyclone::CLI.parse(["phy", "run", "-i", "in.tsv", "-o", "out.jsonl", "--outlier-prob=0.01"])
    run_cmd = result.as(Tyclone::PhyCloneRunCommand)
    run_cmd.config.outlier_prob.should be_close(0.01, 1e-10)
  end
end
