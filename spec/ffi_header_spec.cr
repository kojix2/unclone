require "./spec_helper"

describe "FFI header" do
  it "stays aligned with the current exported kernel ABI" do
    # Normalize line endings so the spec is agnostic to CRLF checkouts (e.g. Windows CI).
    header = File.read(File.expand_path("../rust-kernel/include/pcv_kernel.h", __DIR__)).gsub("\r\n", "\n")
    crystal_ffi = File.read(File.expand_path("../src/unclone/ffi.cr", __DIR__)).gsub("\r\n", "\n")

    header_functions = header.scan(/(?:int|size_t|void|const\s+(?:char|int32_t|double)\*)\s+(pcv_\w+)\s*\(/).map(&.[1]).sort!
    crystal_functions = crystal_ffi.scan(/fun\s+\w+\s*=\s*(pcv_\w+)\s*\(/).map(&.[1]).sort!
    header_functions.should eq(crystal_functions)

    pcv_row_fields = header.match!(/typedef struct \{\n((?:  .+\n)+)\} PcvRow;/)[1].lines.map(&.strip).reject(&.empty?)
    pcv_row_fields.should eq([
      "int32_t mutation_index;",
      "int32_t sample_index;",
      "int32_t ref_counts;",
      "int32_t alt_counts;",
      "int32_t major_cn;",
      "int32_t minor_cn;",
      "int32_t normal_cn;",
      "double tumour_content;",
      "double error_rate;",
    ])

    pcv_config_fields = header.match!(/typedef struct \{\n((?:  .+\n)+)\} PcvConfig;/)[1].lines.map(&.strip).reject(&.empty?)
    pcv_config_fields.should eq([
      "int32_t num_clusters;",
      "int32_t num_grid_points;",
      "int32_t num_restarts;",
      "int32_t max_iters;",
      "int32_t print_freq;",
      "int32_t kernel_threads;",
      "int32_t restart_parallelism;",
      "double convergence_threshold;",
      "double mix_weight_prior;",
      "double precision;",
      "uint8_t density;",
      "uint8_t use_seed;",
      "uint64_t seed;",
    ])

    header.should contain("typedef struct PcvTabularResult PcvTabularResult;")
    header.should contain("typedef PcvTabularResult PcvResult;")
  end
end
