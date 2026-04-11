require "./spec_helper"

describe "FFI header" do
  it "stays aligned with the current exported kernel ABI" do
    header = File.read(File.expand_path("../rust-kernel/include/pcv_kernel.h", __DIR__))

    header.should contain("int32_t restart_parallelism;")
    header.should contain("typedef struct PcvTabularResult PcvTabularResult;")
    header.should contain("int pcv_fit_with_init(")
  end
end
