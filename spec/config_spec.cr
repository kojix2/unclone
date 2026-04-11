require "./spec_helper"

describe Tyclone::ViConfig do
  it "has expected default values" do
    config = Tyclone::ViConfig.new
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
  end
end
