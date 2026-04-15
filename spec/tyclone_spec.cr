require "./spec_helper"

describe Tyclone do
  it "has a version" do
    Tyclone::VERSION.should_not be_empty
  end

  it "parses fit command" do
    config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv"])
    config.command.should eq("fit-vi")
    config.action.should eq(Tyclone::Action::Fit)
    config.in_file.should eq("in.tsv")
    config.out_file.should eq("out.tsv")
  end

  it "parses --help" do
    config = Tyclone::CLI.parse(["--help"])
    config.action.should eq(Tyclone::Action::Help)
    config.help_message.should contain("Usage: tyclone")
  end

  it "parses --version" do
    config = Tyclone::CLI.parse(["--version"])
    config.action.should eq(Tyclone::Action::Version)
  end

  it "raises on missing command" do
    expect_raises(Tyclone::CliError, /Missing command/) do
      Tyclone::CLI.parse([] of String)
    end
  end

  describe "fit-vi command options" do
    it "parses -c / --num-clusters" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "-c", "40"])
      config.num_clusters.should eq(40)
    end

    it "parses -d binomial" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "-d", "binomial"])
      config.density.should eq(Tyclone::Density::Binomial)
    end

    it "parses -d beta-binomial" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "-d", "beta-binomial"])
      config.density.should eq(Tyclone::Density::BetaBinomial)
    end

    it "raises on invalid density name" do
      expect_raises(Tyclone::CliError, /Invalid density/) do
        Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "-d", "gaussian"])
      end
    end

    it "parses -g / --num-grid-points" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "-g", "50"])
      config.num_grid_points.should eq(50)
    end

    it "parses -r / --num-restarts" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "-r", "5"])
      config.num_restarts.should eq(5)
    end

    it "parses --max-iters" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--max-iters=500"])
      config.max_iters.should eq(500)
    end

    it "parses --seed" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--seed=42"])
      config.seed.should eq(42_u64)
    end

    it "parses --precision" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--precision=500.0"])
      config.precision.should eq(500.0)
    end

    it "parses --convergence-threshold" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--convergence-threshold=1e-4"])
      config.convergence_threshold.should be_close(1e-4, 1e-12)
    end

    it "parses --mix-weight-prior" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--mix-weight-prior=2.0"])
      config.mix_weight_prior.should eq(2.0)
    end

    it "parses --kernel-threads" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--kernel-threads=4"])
      config.kernel_threads.should eq(4)
    end

    it "parses --restart-parallelism" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--restart-parallelism=2"])
      config.restart_parallelism.should eq(2)
    end

    it "parses --compress" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--compress"])
      config.compress?.should be_true
    end

    it "parses --print-freq" do
      config = Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--print-freq=50"])
      config.print_freq.should eq(50)
    end

    it "raises when --in-file is omitted" do
      expect_raises(Tyclone::CliError) do
        Tyclone::CLI.parse(["fit-vi", "-o", "out.tsv"])
      end
    end

    it "raises when --out-file is omitted" do
      expect_raises(Tyclone::CliError) do
        Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv"])
      end
    end

    it "raises on unknown option" do
      expect_raises(Tyclone::CliError, /is not a valid option/) do
        Tyclone::CLI.parse(["fit-vi", "-i", "in.tsv", "-o", "out.tsv", "--unknown"])
      end
    end

    it "fit --help shows fit-specific usage" do
      config = Tyclone::CLI.parse(["fit-vi", "--help"])
      config.action.should eq(Tyclone::Action::Help)
      config.help_message.should contain("--in-file")
      config.help_message.should contain("--num-clusters")
    end
  end

  describe "fit-mcmc command options" do
    it "parses fit-mcmc command and sets engine" do
      config = Tyclone::CLI.parse(["fit-mcmc", "-i", "in.tsv", "-o", "out.tsv"])
      config.command.should eq("fit-mcmc")
      config.engine.should eq(Tyclone::Engine::MCMC)
    end

    it "parses MCMC-specific options" do
      config = Tyclone::CLI.parse([
        "fit-mcmc",
        "-i", "in.tsv",
        "-o", "out.tsv",
        "--num-iters=200",
        "--burnin=50",
        "--thin=5",
        "--alpha=2.5",
        "--alpha-prior-shape=1.5",
        "--alpha-prior-rate=0.25",
        "--init-method=connected",
        "--base-measure-alpha=2.0",
        "--base-measure-beta=3.0",
        "--mh-step-size=0.1",
        "--mh-precision-step=0.05",
        "--mh-precision-proposal-precision=0.02",
        "--precision=800.0",
      ])
      config.num_iters.should eq(200)
      config.burnin.should eq(50)
      config.thin.should eq(5)
      config.alpha.should eq(2.5)
      config.alpha_prior_shape.should eq(1.5)
      config.alpha_prior_rate.should eq(0.25)
      config.init_method.should eq("connected")
      config.base_measure_alpha.should eq(2.0)
      config.base_measure_beta.should eq(3.0)
      config.mh_step_size.should eq(0.1)
      config.mh_precision_step.should eq(0.05)
      config.mh_precision_proposal_precision.should eq(0.02)
      config.precision.should eq(800.0)
    end

    it "fit-mcmc --help shows MCMC-specific usage" do
      config = Tyclone::CLI.parse(["fit-mcmc", "--help"])
      config.action.should eq(Tyclone::Action::Help)
      config.help_message.should contain("--num-iters")
      config.help_message.should contain("--alpha")
      config.help_message.should contain("--init-method")
      config.help_message.should contain("--base-measure-alpha")
    end

    it "rejects an invalid init-method" do
      expect_raises(Tyclone::CliError, /Invalid init-method/) do
        Tyclone::CLI.parse(["fit-mcmc", "-i", "in.tsv", "-o", "out.tsv", "--init-method=other"])
      end
    end
  end
end
