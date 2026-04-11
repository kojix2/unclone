require "option_parser"

module Tyclone
  module CLI
    class Parser < OptionParser
      @selected_command : Symbol?
      @show_help : Bool
      @show_version : Bool

      property help_message : String

      macro _on_help_
        on("-h", "--help", "Show this help") do
          @show_help = true
        end

        separator "\n"
        # Keep the current parser help text because OptionParser restores state.
        @help_message = self.to_s
      end

      macro _set_command_(command_name, banner)
        @selected_command = {{ command_name }}
        @handlers.clear
        @flags.clear
        self.banner = "\n{{ banner.id }}\n"
      end

      def initialize
        super()
        @vi_config = ViConfig.new
        @selected_command = nil
        @show_help = false
        @show_version = false
        @help_message = ""

        self.summary_width = 26
        self.banner = <<-BANNER

          Program: #{PROGRAM} (Reimplementation of PyClone-VI and PhyClone)
          Version: #{VERSION}
          Source:  #{SOURCE}

          Usage: tyclone <command> [options]

          BANNER

        separator "Commands:"

        on("phy", "Run PhyClone-compatible workflows") do
        end

        on("vi", "Run PyClone-VI variational inference") do
          _set_command_(:fit_vi, "Usage: tyclone vi [options]")

          on("-i FILE", "--in-file=FILE", "Input TSV") { |v| @vi_config.in_file = v }
          on("-o FILE", "--out-file=FILE", "Output TSV") { |v| @vi_config.out_file = v }
          on("-c N", "--num-clusters=N", "Number of clusters") { |v| @vi_config.num_clusters = v.to_i32 }
          on("-d D", "--density=D", "binomial or beta-binomial") do |v|
            @vi_config.density = case v
                                 when "binomial"      then Density::Binomial
                                 when "beta-binomial" then Density::BetaBinomial
                                 else
                                   raise CliError.new("Invalid density: #{v}")
                                 end
          end
          on("--seed=U", "Random seed") { |v| @vi_config.seed = v.to_u64 }
          on("--compress", "Write gzip output") { @vi_config.compress = true }
          on("-g N", "--num-grid-points=N", "Number of CCF grid points") { |v| @vi_config.num_grid_points = v.to_i32 }
          on("-r N", "--num-restarts=N", "Number of restarts") { |v| @vi_config.num_restarts = v.to_i32 }
          on("--convergence-threshold=F", "Convergence threshold") { |v| @vi_config.convergence_threshold = v.to_f64 }
          on("--max-iters=N", "Maximum VI iterations") { |v| @vi_config.max_iters = v.to_i32 }
          on("--mix-weight-prior=F", "Mixture weight prior") { |v| @vi_config.mix_weight_prior = v.to_f64 }
          on("--precision=F", "Beta-binomial precision") { |v| @vi_config.precision = v.to_f64 }
          on("--print-freq=N", "Progress print interval") { |v| @vi_config.print_freq = v.to_i32 }
          on("--kernel-threads=N", "Kernel threads (0=auto)") { |v| @vi_config.kernel_threads = v.to_i32 }
          on("--restart-parallelism=N", "Outer restart parallelism") { |v| @vi_config.restart_parallelism = v.to_i32 }
          on("--python-compatible", "Use Python/NumPy-compatible initialization (requires numpy)") { @vi_config.python_compatible = true }

          _on_help_
        end

        separator

        on("-v", "--version", "Show version") do
          @show_version = true
        end
        _on_help_

        invalid_option do |flag|
          raise CliError.new("#{flag} is not a valid option")
        end

        missing_option do |flag|
          raise CliError.new("#{flag} option expects an argument")
        end
      end

      def parse(args : Array(String)) : Command
        if args.first? == "phy"
          return parse_phy(args[1..])
        end

        missing_command = args.empty?
        super(args)

        if @show_help
          return HelpCommand.new(help_message.empty? ? to_s : help_message)
        end

        return VersionCommand.new if @show_version

        raise CliError.new("Missing command") if missing_command || @selected_command.nil?

        case @selected_command
        when :fit_vi
          validate_required_options(@vi_config.in_file, @vi_config.out_file)
          FitViCommand.new(@vi_config)
        else
          raise CliError.new("Missing command")
        end
      end

      private def validate_required_options(in_file : String, out_file : String)
        if in_file.empty? || out_file.empty?
          raise CliError.new("Both --in-file and --out-file are required")
        end
      end

      private def parse_phy(args : Array(String)) : Command
        return HelpCommand.new(phy_help_message) if args.empty?

        case args.first
        when "-h", "--help"
          HelpCommand.new(phy_help_message)
        when "run"
          parse_phy_run(args[1..])
        when "map"
          parse_phy_map(args[1..])
        when "consensus"
          parse_phy_consensus(args[1..])
        when "topology-report"
          parse_phy_topology_report(args[1..])
        else
          raise CliError.new("Unknown phy subcommand: #{args.first}")
        end
      end

      private def parse_phy_trace_io_command(
        args : Array(String),
        banner : String,
        config : PhyCloneMapConfig | PhyCloneConsensusConfig | PhyCloneTopologyReportConfig,
      ) : Command
        show_help = false
        parser = OptionParser.new do |opts|
          opts.banner = banner
          opts.on("-i FILE", "--in-file=FILE", "Input JSONL trace") { |v| config.in_file = v }
          opts.on("-o FILE", "--out-file=FILE", "Output path") { |v| config.out_file = v }
          opts.on("-h", "--help", "Show this help") { show_help = true }
        end
        help = parser.to_s
        parser.parse(args)
        return HelpCommand.new(help) if show_help
        validate_required_options(config.in_file, config.out_file)
        build_phy_trace_io_command(config)
      rescue ex : OptionParser::Exception
        raise CliError.new(ex.message || "Invalid phy command options")
      end

      private def build_phy_trace_io_command(config : PhyCloneMapConfig) : Command
        PhyCloneMapCommand.new(config)
      end

      private def build_phy_trace_io_command(config : PhyCloneConsensusConfig) : Command
        PhyCloneConsensusCommand.new(config)
      end

      private def build_phy_trace_io_command(config : PhyCloneTopologyReportConfig) : Command
        PhyCloneTopologyReportCommand.new(config)
      end

      private def phy_help_message : String
        <<-HELP

          Usage: tyclone phy <subcommand> [options]

          Status:
            In progress.
            Compatibility with upstream PhyClone is being implemented step by step.

          Subcommands:
            run               Run the initial PhyClone trace writer
            map               Build a point estimate from a trace
            consensus         Build a consensus tree from a trace
            topology-report   Summarize sampled topologies from a trace

          Run `tyclone phy <subcommand> --help` for subcommand options.

          HELP
      end

      private def parse_phy_run(args : Array(String)) : Command
        config = PhyCloneRunConfig.new
        show_help = false
        parser = OptionParser.new do |opts|
          opts.banner = "Usage: tyclone phy run [options]"
          opts.on("-i FILE", "--in-file=FILE", "Input TSV") { |v| config.in_file = v }
          opts.on("-o FILE", "--out-file=FILE", "Output JSONL trace") { |v| config.out_file = v }
          opts.on("-c FILE", "--cluster-file=FILE", "Optional cluster assignment TSV (mutation_id, cluster_id)") { |v| config.cluster_file = v }
          opts.on("-n N", "--num-iters=N", "Number of main-chain MCMC transitions") { |v| config.num_iters = v.to_i32 }
          opts.on("--num-chains=N", "Number of chains") { |v| config.num_chains = v.to_i32 }
          opts.on("--num-samples-data-point=N", "Data-point Gibbs updates per iteration") { |v| config.num_samples_data_point = v.to_i32 }
          opts.on("--num-samples-prune-regraph=N", "Prune-regraph updates per iteration") { |v| config.num_samples_prune_regraft = v.to_i32 }
          opts.on("-s F", "--subtree-update-prob=F", "Probability of subtree update") { |v| config.subtree_update_prob = v.to_f64 }
          opts.on("--num-particles=N", "Number of SMC particles (>= 2 required)") { |v| config.num_particles = v.to_i32 }
          opts.on("-b N", "--burnin=N", "Number of SMC burn-in iterations") { |v| config.burn_in_iters = v.to_i32 }
          opts.on("-t F", "--max-time=F", "Maximum runtime in seconds for burn-in + main MCMC") { |v| config.max_time = v.to_f64 }
          opts.on("--print-freq=N", "Progress print interval for compatible MCMC") { |v| config.print_freq = v.to_i32 }
          opts.on("--thin=N", "Record every N-th main-chain transition") { |v| config.thin = v.to_i32 }
          opts.on("--resample-threshold=F", "SMC resampling threshold in [0, 1]") { |v| config.resample_threshold = v.to_f64 }
          opts.on("--concentration-update", "Update concentration parameter during MCMC") { config.concentration_update = true }
          opts.on("--no-concentration-update", "Disable concentration parameter updates") { config.concentration_update = false }
          opts.on("--concentration-value=F", "Initial concentration parameter value") { |v| config.concentration_value = v.to_f64 }
          opts.on("-p NAME", "--proposal=NAME", "bootstrap, fully-adapted, or semi-adapted") do |v|
            config.proposal =
              case v
              when "bootstrap"     then PhyCloneProposal::Bootstrap
              when "fully-adapted" then PhyCloneProposal::FullyAdapted
              when "semi-adapted"  then PhyCloneProposal::SemiAdapted
              else
                raise CliError.new("Invalid proposal: #{v}")
              end
          end
          opts.on("-d D", "--density=D", "binomial or beta-binomial for outlier exact model") do |v|
            config.density = case v
                             when "binomial"      then Density::Binomial
                             when "beta-binomial" then Density::BetaBinomial
                             else
                               raise CliError.new("Invalid density: #{v}")
                             end
          end
          opts.on("--grid-size=N", "Outlier exact-model CCF grid points") { |v| config.num_grid_points = v.to_i32 }
          opts.on("--precision=F", "Outlier exact-model beta-binomial precision") { |v| config.precision = v.to_f64 }
          opts.on("--assign-loss-prob", "Assign loss probability by chromosome") { config.assign_loss_prob = true }
          opts.on("--user-provided-loss-prob", "Use loss_prob column from input TSV") { config.user_provided_loss_prob = true }
          opts.on("--loss-prob=F", "Default loss probability for non-high-loss chromosomes") { |v| config.loss_prob = v.to_f64 }
          opts.on("--high-loss-prob=F", "Loss probability for high-loss chromosomes") { |v| config.high_loss_prob = v.to_f64 }
          opts.on("-l F", "--outlier-prob=F", "Global fallback outlier probability (default: 0.0)") { |v| config.outlier_prob = v.to_f64 }
          opts.on("--seed=U", "Random seed") { |v| config.seed = v.to_u64 }
          opts.on("--compress", "Write gzip output") { config.compress = true }
          opts.on("-h", "--help", "Show this help") { show_help = true }
          opts.invalid_option do |flag|
            flag_name = flag.includes?('=') ? flag.split('=').first : flag
            raise CliError.new("#{flag_name} is not a valid option")
          end
        end
        help = parser.to_s
        parser.parse(args)
        return HelpCommand.new(help) if show_help
        validate_required_options(config.in_file, config.out_file)
        validate_phy_run_options(config)
        PhyCloneRunCommand.new(config)
      rescue ex : OptionParser::Exception
        raise CliError.new(ex.message || "Invalid phy run options")
      end

      private def validate_phy_run_options(config : PhyCloneRunConfig)
        validate_phy_loss_options(config)
        raise CliError.new("--num-samples-data-point must be >= 0") if config.num_samples_data_point < 0
        raise CliError.new("--num-samples-prune-regraph must be >= 0") if config.num_samples_prune_regraft < 0
        unless (0.0..1.0).includes?(config.subtree_update_prob)
          raise CliError.new("--subtree-update-prob must be within [0, 1]")
        end
        raise CliError.new("--num-particles must be >= 2") if config.num_particles < 2
        raise CliError.new("--burnin must be >= 0") if config.burn_in_iters < 0
        validate_phy_time_options(config)
        raise CliError.new("--thin must be >= 1") if config.thin < 1
        unless (0.0..1.0).includes?(config.resample_threshold)
          raise CliError.new("--resample-threshold must be within [0, 1]")
        end
        raise CliError.new("--concentration-value must be > 0") unless config.concentration_value > 0.0
      end

      private def validate_phy_time_options(config : PhyCloneRunConfig)
        if !config.max_time.finite? && !config.max_time.infinite?
          raise CliError.new("--max-time must be finite and >= 0, or Infinity")
        end
        if config.max_time.finite? && config.max_time < 0.0
          raise CliError.new("--max-time must be >= 0")
        end
        raise CliError.new("--print-freq must be >= 1") if config.print_freq < 1
      end

      private def validate_phy_loss_options(config : PhyCloneRunConfig)
        if config.assign_loss_prob? && config.user_provided_loss_prob?
          raise CliError.new("--assign-loss-prob and --user-provided-loss-prob are mutually exclusive")
        end

        if (config.assign_loss_prob? || config.user_provided_loss_prob?) && config.cluster_file.nil?
          raise CliError.new("--assign-loss-prob and --user-provided-loss-prob require --cluster-file")
        end

        unless (0.0..1.0).includes?(config.loss_prob) && (0.0..1.0).includes?(config.high_loss_prob)
          raise CliError.new("--loss-prob and --high-loss-prob must be within [0, 1]")
        end

        if config.assign_loss_prob? && !(config.high_loss_prob > config.loss_prob)
          raise CliError.new("--high-loss-prob must be greater than --loss-prob when --assign-loss-prob is enabled")
        end
      end

      private def parse_phy_map(args : Array(String)) : Command
        parse_phy_trace_io_command(args, "Usage: tyclone phy map [options]", PhyCloneMapConfig.new)
      end

      private def parse_phy_consensus(args : Array(String)) : Command
        config = PhyCloneConsensusConfig.new
        show_help = false
        parser = OptionParser.new do |opts|
          opts.banner = "Usage: tyclone phy consensus [options]"
          opts.on("-i FILE", "--in-file=FILE", "Input JSONL trace") { |v| config.in_file = v }
          opts.on("-o FILE", "--out-file=FILE", "Output consensus JSON") { |v| config.out_file = v }
          opts.on("--consensus-threshold=F", "Minimum topology support fraction") { |v| config.consensus_threshold = v.to_f64 }
          opts.on("--weight=MODE", "Consensus weighting mode: counts or log_p") { |v| config.weight = v }
          opts.on("-h", "--help", "Show this help") { show_help = true }
        end
        help = parser.to_s
        parser.parse(args)
        return HelpCommand.new(help) if show_help
        validate_required_options(config.in_file, config.out_file)
        unless (0.0..1.0).includes?(config.consensus_threshold)
          raise CliError.new("--consensus-threshold must be within [0, 1]")
        end
        unless {"counts", "log_p"}.includes?(config.weight)
          raise CliError.new("--weight must be one of: counts, log_p")
        end
        PhyCloneConsensusCommand.new(config)
      rescue ex : OptionParser::Exception
        raise CliError.new(ex.message || "Invalid phy consensus options")
      end

      private def parse_phy_topology_report(args : Array(String)) : Command
        parse_phy_trace_io_command(args, "Usage: tyclone phy topology-report [options]", PhyCloneTopologyReportConfig.new)
      end
    end

    def self.parse(args : Array(String)) : Command
      Parser.new.parse(args)
    end
  end
end
