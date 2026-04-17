require "option_parser"

module Tyclone
  module CLI
    class Parser < OptionParser
      getter opt : Config
      property help_message : String

      macro _on_help_
        on("-h", "--help", "Show this help") do
          opt.action = Action::Help
        end

        separator "\n"
        # Keep the current parser help text because OptionParser restores state.
        @help_message = self.to_s
      end

      macro _set_action_(action, banner)
        opt.action = {{ action }}
        @handlers.clear
        @flags.clear
        self.banner = "\n{{ banner.id }}\n"
      end

      private def add_common_fit_options
        on("-i FILE", "--in-file=FILE", "Input TSV") { |v| opt.in_file = v }
        on("-o FILE", "--out-file=FILE", "Output TSV") { |v| opt.out_file = v }
        on("-c N", "--num-clusters=N", "Number of clusters") { |v| opt.num_clusters = v.to_i32 }
        on("-d D", "--density=D", "binomial or beta-binomial") do |v|
          opt.density = case v
                        when "binomial"      then Density::Binomial
                        when "beta-binomial" then Density::BetaBinomial
                        else
                          raise CliError.new("Invalid density: #{v}")
                        end
        end
        on("--seed=U", "Random seed") { |v| opt.seed = v.to_u64 }
        on("--compress", "Write gzip output") { opt.compress = true }
      end

      private def parse_init_method(value : String) : String
        case value
        when "connected", "disconnected"
          value
        else
          raise CliError.new("Invalid init-method: #{value}")
        end
      end

      def initialize
        super()
        @opt = Config.new
        @help_message = ""

        self.summary_width = 26
        self.banner = <<-BANNER

          Program: #{PROGRAM} (Reimplementation of PyClone)
          Version: #{VERSION}
          Source:  #{SOURCE}

          Usage: tyclone <command> [options]

          BANNER

        separator "Commands:"

        on("fit-vi", "Run PyClone-VI variational inference") do
          _set_action_(Action::Fit, "Usage: tyclone fit-vi [options]")
          opt.command = "fit-vi"
          opt.engine = Engine::VI

          add_common_fit_options
          on("-g N", "--num-grid-points=N", "Number of CCF grid points") { |v| opt.num_grid_points = v.to_i32 }
          on("-r N", "--num-restarts=N", "Number of restarts") { |v| opt.num_restarts = v.to_i32 }
          on("--convergence-threshold=F", "Convergence threshold") { |v| opt.convergence_threshold = v.to_f64 }
          on("--max-iters=N", "Maximum VI iterations") { |v| opt.max_iters = v.to_i32 }
          on("--mix-weight-prior=F", "Mixture weight prior") { |v| opt.mix_weight_prior = v.to_f64 }
          on("--precision=F", "Beta-binomial precision") { |v| opt.precision = v.to_f64 }
          on("--print-freq=N", "Progress print interval") { |v| opt.print_freq = v.to_i32 }
          on("--kernel-threads=N", "Kernel threads (0=auto)") { |v| opt.kernel_threads = v.to_i32 }
          on("--restart-parallelism=N", "Outer restart parallelism") { |v| opt.restart_parallelism = v.to_i32 }
          on("--python-compatible", "Use Python/NumPy-compatible initialization (requires numpy)") { opt.python_compatible = true }

          _on_help_
        end

        on("fit-mcmc", "Run PyClone MCMC inference") do
          _set_action_(Action::Fit, "Usage: tyclone fit-mcmc [options]")
          opt.command = "fit-mcmc"
          opt.engine = Engine::MCMC

          add_common_fit_options
          on("--precision=F", "Beta-binomial precision (0=adaptive)") { |v| opt.precision = v.to_f64 }
          on("--num-iters=N", "Total MCMC iterations before burn-in/thinning") { |v| opt.num_iters = v.to_i32 }
          on("--burnin=N", "Number of saved trace rows to discard from the start") { |v| opt.burnin = v.to_i32 }
          on("--thin=N", "Keep every N-th saved trace row after burn-in") { |v| opt.thin = v.to_i32 }
          on("--alpha=F", "CRP concentration parameter") { |v| opt.alpha = v.to_f64 }
          on("--alpha-prior-shape=F", "Gamma prior shape for alpha") { |v| opt.alpha_prior_shape = v.to_f64 }
          on("--alpha-prior-rate=F", "Gamma prior rate for alpha") { |v| opt.alpha_prior_rate = v.to_f64 }
          on("--init-method=METHOD", "connected or disconnected") { |v| opt.init_method = parse_init_method(v) }
          on("--base-measure-alpha=F", "Beta base measure alpha") { |v| opt.base_measure_alpha = v.to_f64 }
          on("--base-measure-beta=F", "Beta base measure beta") { |v| opt.base_measure_beta = v.to_f64 }
          on("--mh-step-size=F", "Atom MH step size") { |v| opt.mh_step_size = v.to_f64 }
          on("--mh-precision-step=F", "Precision MH step size (0=fixed)") { |v| opt.mh_precision_step = v.to_f64 }
          on("--mh-precision-proposal-precision=F", "Gamma proposal precision for beta-binomial precision") { |v| opt.mh_precision_proposal_precision = v.to_f64 }
          on("--print-freq=N", "Print progress every N iters (0=silent)") { |v| opt.print_freq = v.to_i32 }

          _on_help_
        end

        separator

        on("-v", "--version", "Show version") do
          opt.action = Action::Version
        end
        _on_help_

        invalid_option do |flag|
          raise CliError.new("#{flag} is not a valid option")
        end

        missing_option do |flag|
          raise CliError.new("#{flag} option expects an argument")
        end
      end

      def parse(args : Array(String)) : Config
        raise CliError.new("Missing command") if args.empty?

        super(args)

        if opt.action == Action::Help
          opt.help_message = help_message.empty? ? to_s : help_message
          return opt
        end

        return opt unless opt.action == Action::Fit
        if opt.in_file.empty? || opt.out_file.empty?
          raise CliError.new("Both --in-file and --out-file are required")
        end

        opt
      end
    end

    def self.parse(args : Array(String)) : Config
      Parser.new.parse(args)
    end
  end
end
