module Tyclone
  enum Density
    Binomial
    BetaBinomial
  end

  enum Action
    Fit
    Help
    Version
  end

  enum Engine
    VI
    MCMC
  end

  class Config
    property action : Action
    property command : String
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
    property help_message : String
    property engine : Engine
    property num_iters : Int32
    property burnin : Int32
    property thin : Int32
    property alpha : Float64
    property alpha_prior_shape : Float64
    property alpha_prior_rate : Float64
    property init_method : String
    property base_measure_alpha : Float64
    property base_measure_beta : Float64
    property mh_step_size : Float64
    property mh_precision_step : Float64
    property mh_precision_proposal_precision : Float64

    def initialize
      @action = Action::Fit
      @command = "fit-vi"
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
      @help_message = ""
      @engine = Engine::VI
      @num_iters = 1000
      @burnin = 0
      @thin = 1
      @alpha = 1.0
      @alpha_prior_shape = 1.0
      @alpha_prior_rate = 0.001
      @init_method = "disconnected"
      @base_measure_alpha = 1.0
      @base_measure_beta = 1.0
      @mh_step_size = 0.01
      @mh_precision_step = 0.0
      @mh_precision_proposal_precision = 0.01
    end
  end
end
