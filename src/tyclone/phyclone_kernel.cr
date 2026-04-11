module Tyclone
  module PhyCloneKernel
    def self.generate_trace(
      rows : Array(PhyClone::InputRow),
      cluster_file_rows : Array(PhyClone::Input::ClusterRow)?,
      config : PhyCloneRunConfig,
      num_chains : Int32,
      num_iters : Int32,
      seed : UInt64?,
    ) : String
      cluster_json = build_request_json(rows, cluster_file_rows, config)
      out_json = Pointer(UInt8).null
      error_ptr = Pointer(KernelAbi::Error).null

      rc = LibPcv.pcv_phyclone_generate_trace(
        cluster_json.to_unsafe,
        num_chains,
        num_iters,
        seed.nil? ? 0_u8 : 1_u8,
        seed || 0_u64,
        pointerof(out_json),
        pointerof(error_ptr)
      )

      if rc != 0
        message = "Unknown kernel error"
        unless error_ptr.null?
          message_ptr = LibPcv.pcv_error_message(error_ptr)
          message = String.new(message_ptr) unless message_ptr.null?
          LibPcv.pcv_error_free(error_ptr)
        end
        raise KernelError.new(message)
      end

      begin
        String.new(out_json)
      ensure
        LibPcv.pcv_string_free(out_json) unless out_json.null?
      end
    end

    private def self.build_request_json(
      rows : Array(PhyClone::InputRow),
      cluster_file_rows : Array(PhyClone::Input::ClusterRow)?,
      config : PhyCloneRunConfig,
    ) : String
      String.build do |io|
        JSON.build(io) do |json|
          json.object do
            json.field "rows" do
              json.array do
                rows.each do |row|
                  json.object do
                    json.field "mutation_id", row.mutation_id
                    json.field "sample_id", row.sample_id
                    json.field "ref_counts", row.ref_counts
                    json.field "alt_counts", row.alt_counts
                    json.field "major_cn", row.major_cn
                    json.field "minor_cn", row.minor_cn
                    json.field "normal_cn", row.normal_cn
                    json.field "tumour_content", row.tumour_content
                    json.field "error_rate", row.error_rate
                    json.field "chrom", row.chrom
                    json.field "loss_prob", row.loss_prob
                    json.field "outlier_prob", row.outlier_prob
                    if cid = row.cluster_id
                      json.field "cluster_id", cid.to_s
                    else
                      json.field "cluster_id", nil
                    end
                  end
                end
              end
            end
            json.field "cluster_rows" do
              json.array do
                cluster_file_rows.try &.each do |cluster_row|
                  json.object do
                    json.field "mutation_id", cluster_row.mutation_id
                    json.field "cluster_id", cluster_row.cluster_id
                    json.field "sample_id", cluster_row.sample_id
                    json.field "chrom", cluster_row.chrom
                    json.field "cellular_prevalence", cluster_row.cellular_prevalence
                    json.field "outlier_prob", cluster_row.outlier_prob
                  end
                end
              end
            end
            json.field "options" do
              json.object do
                json.field "density_code", config.density == Density::Binomial ? 0 : 1
                json.field "precision", config.precision
                json.field "grid_size", config.num_grid_points
                json.field "outlier_prob", config.outlier_prob
                json.field "num_particles", config.num_particles
                json.field "burnin", config.burn_in_iters
                max_time_val = config.max_time.infinite? ? nil : config.max_time
                json.field "max_time", max_time_val
                json.field "print_freq", config.print_freq
                json.field "thin", config.thin
                json.field "resample_threshold", config.resample_threshold
                json.field "use_phyclone_mcmc", true
                # PhyClone-compatible defaults:
                # 0=bootstrap, 1=fully-adapted, 2=semi-adapted
                proposal_code = case config.proposal
                                when PhyCloneProposal::Bootstrap
                                  0
                                when PhyCloneProposal::FullyAdapted
                                  1
                                when PhyCloneProposal::SemiAdapted
                                  2
                                end
                json.field "proposal_code", proposal_code
                json.field "num_samples_data_point", config.num_samples_data_point
                json.field "num_samples_prune_regraft", config.num_samples_prune_regraft
                json.field "subtree_update_prob", config.subtree_update_prob
                json.field "concentration_update", config.concentration_update?
                json.field "concentration_value", config.concentration_value
                json.field "assign_loss_prob", config.assign_loss_prob?
                json.field "user_provided_loss_prob", config.user_provided_loss_prob?
                json.field "loss_prob", config.loss_prob
                json.field "high_loss_prob", config.high_loss_prob
              end
            end
          end
        end
      end
    end
  end
end
