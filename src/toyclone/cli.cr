require "option_parser"

module Toyclone
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

      def initialize
        super()
        @opt = Config.new
        @help_message = ""

        self.summary_width = 26
        self.banner = <<-BANNER

          Program: #{PROGRAM} (Reimplementation of PyClone-VI)
          Version: #{VERSION}
          Source:  #{SOURCE}

          Usage: toyclone <command> [options]

          BANNER

        separator "Commands:"

        on("fit", "Run PyClone-VI style inference") do
          _set_action_(Action::Fit, "Usage: toyclone fit [options]")
          opt.command = "fit"

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
          on("-g N", "--num-grid-points=N", "Number of grid points") { |v| opt.num_grid_points = v.to_i32 }
          on("-r N", "--num-restarts=N", "Number of restarts") { |v| opt.num_restarts = v.to_i32 }
          on("--convergence-threshold=F", "Convergence threshold") { |v| opt.convergence_threshold = v.to_f64 }
          on("--max-iters=N", "Maximum VI iterations") { |v| opt.max_iters = v.to_i32 }
          on("--mix-weight-prior=F", "Mixture weight prior") { |v| opt.mix_weight_prior = v.to_f64 }
          on("--precision=F", "Beta-binomial precision") { |v| opt.precision = v.to_f64 }
          on("--print-freq=N", "Progress print interval") { |v| opt.print_freq = v.to_i32 }
          on("--seed=U", "Random seed") { |v| opt.seed = v.to_u64 }
          on("--kernel-threads=N", "Kernel threads") { |v| opt.kernel_threads = v.to_i32 }
          on("--restart-parallelism=N", "Outer restart parallelism") { |v| opt.restart_parallelism = v.to_i32 }
          on("--compress", "Write gzip output") { opt.compress = true }

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
