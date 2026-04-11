module Tyclone
  enum Density
    Binomial
    BetaBinomial
  end

  enum PhyCloneProposal
    Bootstrap
    FullyAdapted
    SemiAdapted
  end

  class ViConfig
    property in_file : String
    property out_file : String
    property num_clusters : Int32
    property density : Density
    property num_grid_points : Int32
    property num_restarts : Int32
    property convergence_threshold : Float64
    property max_iters : Int32
    property mix_weight_prior : Float64
    property precision : Float64
    property print_freq : Int32
    property seed : UInt64?
    property kernel_threads : Int32
    property restart_parallelism : Int32
    property? python_compatible : Bool
    property? compress : Bool

    def initialize
      @in_file = ""
      @out_file = ""
      @num_clusters = 10
      @density = Density::Binomial
      @num_grid_points = 100
      @num_restarts = 1
      @convergence_threshold = 1e-6
      @max_iters = 10_000
      @mix_weight_prior = 1.0
      @precision = 200.0
      @print_freq = 100
      @seed = nil
      @kernel_threads = 0
      @restart_parallelism = 1
      @python_compatible = false
      @compress = false
    end
  end

  class PhyCloneRunConfig
    property in_file : String
    property out_file : String
    property cluster_file : String?
    property num_iters : Int32
    property num_chains : Int32
    property num_samples_data_point : Int32
    property num_samples_prune_regraft : Int32
    property subtree_update_prob : Float64
    property num_particles : Int32
    property burn_in_iters : Int32
    property max_time : Float64
    property print_freq : Int32
    property thin : Int32
    property resample_threshold : Float64
    property? concentration_update : Bool
    property concentration_value : Float64
    property proposal : PhyCloneProposal
    property density : Density
    property num_grid_points : Int32
    property precision : Float64
    property? assign_loss_prob : Bool
    property? user_provided_loss_prob : Bool
    property loss_prob : Float64
    property high_loss_prob : Float64
    property outlier_prob : Float64
    property seed : UInt64?
    property? compress : Bool

    def initialize
      @in_file = ""
      @out_file = ""
      @cluster_file = nil
      @num_iters = 10000
      @num_chains = 1
      @num_samples_data_point = 1
      @num_samples_prune_regraft = 1
      @subtree_update_prob = 0.0
      @num_particles = 100
      @burn_in_iters = 1000
      @max_time = Float64::INFINITY
      @print_freq = 100
      @thin = 1
      @resample_threshold = 0.5
      @concentration_update = true
      @concentration_value = 1.0
      @proposal = PhyCloneProposal::SemiAdapted
      @density = Density::BetaBinomial
      @num_grid_points = 101
      @precision = 400.0
      @assign_loss_prob = false
      @user_provided_loss_prob = false
      @loss_prob = 0.0
      @high_loss_prob = 0.4
      @outlier_prob = 0.0_f64
      @seed = nil
      @compress = false
    end
  end

  class PhyCloneMapConfig
    property in_file : String
    property out_file : String

    def initialize
      @in_file = ""
      @out_file = ""
    end
  end

  class PhyCloneConsensusConfig
    property in_file : String
    property out_file : String
    property consensus_threshold : Float64
    property weight : String

    def initialize
      @in_file = ""
      @out_file = ""
      @consensus_threshold = 0.5
      @weight = "counts"
    end
  end

  class PhyCloneTopologyReportConfig
    property in_file : String
    property out_file : String

    def initialize
      @in_file = ""
      @out_file = ""
    end
  end

  struct HelpCommand
    getter help_message : String

    def initialize(@help_message : String)
    end
  end

  struct VersionCommand
  end

  struct FitViCommand
    getter config : ViConfig

    def initialize(@config : ViConfig)
    end
  end

  struct PhyCloneRunCommand
    getter config : PhyCloneRunConfig

    def initialize(@config : PhyCloneRunConfig)
    end
  end

  struct PhyCloneMapCommand
    getter config : PhyCloneMapConfig

    def initialize(@config : PhyCloneMapConfig)
    end
  end

  struct PhyCloneConsensusCommand
    getter config : PhyCloneConsensusConfig

    def initialize(@config : PhyCloneConsensusConfig)
    end
  end

  struct PhyCloneTopologyReportCommand
    getter config : PhyCloneTopologyReportConfig

    def initialize(@config : PhyCloneTopologyReportConfig)
    end
  end

  alias Command = HelpCommand | VersionCommand | FitViCommand | PhyCloneRunCommand | PhyCloneMapCommand | PhyCloneConsensusCommand | PhyCloneTopologyReportCommand
end
