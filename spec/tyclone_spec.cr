require "./spec_helper"

describe Tyclone do
  it "has a version" do
    Tyclone::VERSION.should_not be_empty
  end

  it "parses fit command" do
    command = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv"])
    command.should be_a(Tyclone::FitViCommand)
    config = command.as(Tyclone::FitViCommand).config
    config.in_file.should eq("in.tsv")
    config.out_file.should eq("out.tsv")
  end

  it "parses --help" do
    command = Tyclone::CLI.parse(["--help"])
    command.should be_a(Tyclone::HelpCommand)
    command.as(Tyclone::HelpCommand).help_message.should contain("Usage: tyclone")
  end

  it "parses --version" do
    Tyclone::CLI.parse(["--version"]).should be_a(Tyclone::VersionCommand)
  end

  it "parses phy --help" do
    command = Tyclone::CLI.parse(["phy", "--help"])
    command.should be_a(Tyclone::HelpCommand)
    command.as(Tyclone::HelpCommand).help_message.should contain("topology-report")
  end

  it "raises on missing command" do
    expect_raises(Tyclone::CliError, /Missing command/) do
      Tyclone::CLI.parse([] of String)
    end
  end

  describe "vi command options" do
    it "parses -c / --num-clusters" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "-c", "40"]).as(Tyclone::FitViCommand).config
      config.num_clusters.should eq(40)
    end

    it "parses -d binomial" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "-d", "binomial"]).as(Tyclone::FitViCommand).config
      config.density.should eq(Tyclone::Density::Binomial)
    end

    it "parses -d beta-binomial" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "-d", "beta-binomial"]).as(Tyclone::FitViCommand).config
      config.density.should eq(Tyclone::Density::BetaBinomial)
    end

    it "raises on invalid density name" do
      expect_raises(Tyclone::CliError, /Invalid density/) do
        Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "-d", "gaussian"])
      end
    end

    it "parses -g / --num-grid-points" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "-g", "50"]).as(Tyclone::FitViCommand).config
      config.num_grid_points.should eq(50)
    end

    it "parses -r / --num-restarts" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "-r", "5"]).as(Tyclone::FitViCommand).config
      config.num_restarts.should eq(5)
    end

    it "parses --max-iters" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--max-iters=500"]).as(Tyclone::FitViCommand).config
      config.max_iters.should eq(500)
    end

    it "parses --seed" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--seed=42"]).as(Tyclone::FitViCommand).config
      config.seed.should eq(42_u64)
    end

    it "parses --precision" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--precision=500.0"]).as(Tyclone::FitViCommand).config
      config.precision.should eq(500.0)
    end

    it "parses --convergence-threshold" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--convergence-threshold=1e-4"]).as(Tyclone::FitViCommand).config
      config.convergence_threshold.should be_close(1e-4, 1e-12)
    end

    it "parses --mix-weight-prior" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--mix-weight-prior=2.0"]).as(Tyclone::FitViCommand).config
      config.mix_weight_prior.should eq(2.0)
    end

    it "parses --kernel-threads" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--kernel-threads=4"]).as(Tyclone::FitViCommand).config
      config.kernel_threads.should eq(4)
    end

    it "parses --restart-parallelism" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--restart-parallelism=2"]).as(Tyclone::FitViCommand).config
      config.restart_parallelism.should eq(2)
    end

    it "parses --compress" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--compress"]).as(Tyclone::FitViCommand).config
      config.compress?.should be_true
    end

    it "parses --print-freq" do
      config = Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--print-freq=50"]).as(Tyclone::FitViCommand).config
      config.print_freq.should eq(50)
    end

    it "raises when --in-file is omitted" do
      expect_raises(Tyclone::CliError) do
        Tyclone::CLI.parse(["vi", "-o", "out.tsv"])
      end
    end

    it "raises when --out-file is omitted" do
      expect_raises(Tyclone::CliError) do
        Tyclone::CLI.parse(["vi", "-i", "in.tsv"])
      end
    end

    it "raises on unknown option" do
      expect_raises(Tyclone::CliError, /is not a valid option/) do
        Tyclone::CLI.parse(["vi", "-i", "in.tsv", "-o", "out.tsv", "--unknown"])
      end
    end

    it "fit --help shows fit-specific usage" do
      command = Tyclone::CLI.parse(["vi", "--help"])
      command.should be_a(Tyclone::HelpCommand)
      command.as(Tyclone::HelpCommand).help_message.should contain("--in-file")
      command.as(Tyclone::HelpCommand).help_message.should contain("--num-clusters")
    end
  end
end
