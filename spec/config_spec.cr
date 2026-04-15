require "./spec_helper"

describe Tyclone::Config do
  it "has expected default values" do
    config = Tyclone::Config.new
    config.action.should eq(Tyclone::Action::Fit)
    config.command.should eq("fit-vi")
    config.in_file.should eq("")
    config.out_file.should eq("")
    config.num_clusters.should eq(10)
    config.density.should eq(Tyclone::Density::Binomial)
    config.num_grid_points.should eq(100)
    config.num_restarts.should eq(1)
    config.convergence_threshold.should be_close(1e-6, 1e-15)
    config.max_iters.should eq(10_000)
    config.mix_weight_prior.should eq(1.0)
    config.precision.should eq(200.0)
    config.print_freq.should eq(100)
    config.seed.should be_nil
    config.kernel_threads.should eq(0)
    config.restart_parallelism.should eq(1)
    config.compress?.should be_false
    config.help_message.should eq("")
  end

  it "has expected MCMC default values" do
    config = Tyclone::Config.new
    config.engine.should eq(Tyclone::Engine::VI)
    config.num_iters.should eq(1000)
    config.burnin.should eq(0)
    config.thin.should eq(1)
    config.alpha.should eq(1.0)
    config.alpha_prior_shape.should eq(1.0)
    config.alpha_prior_rate.should be_close(0.001, 1e-15)
    config.init_method.should eq("disconnected")
    config.base_measure_alpha.should eq(1.0)
    config.base_measure_beta.should eq(1.0)
    config.mh_step_size.should be_close(0.01, 1e-15)
    config.mh_precision_step.should eq(0.0)
    config.mh_precision_proposal_precision.should eq(0.01)
  end
end
