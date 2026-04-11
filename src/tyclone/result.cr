module Tyclone
  struct OutputRow
    getter mutation_id : String
    getter sample_id : String
    getter cluster_id : Int32
    getter cellular_prevalence : Float64
    getter cellular_prevalence_std : Float64
    getter cluster_assignment_prob : Float64

    def initialize(
      @mutation_id : String,
      @sample_id : String,
      @cluster_id : Int32,
      @cellular_prevalence : Float64,
      @cellular_prevalence_std : Float64,
      @cluster_assignment_prob : Float64,
    )
    end
  end

  module ResultBuilder
    def self.build(indexed : IndexedRows, result : TabularKernelResult) : Array(OutputRow)
      cluster_ids = result.mutation_cluster_ids
      cluster_probs = result.mutation_cluster_probs
      prevalence = result.mutation_sample_prevalence
      prevalence_std = result.mutation_sample_prevalence_std
      sample_count = indexed.num_samples

      rows = [] of OutputRow
      indexed.mutation_ids.each_with_index do |mutation_id, m_i|
        cluster_id = cluster_ids[m_i]
        prob = cluster_probs[m_i]
        mutation_offset = m_i * sample_count

        indexed.sample_ids.each_with_index do |sample_id, s_i|
          rows << OutputRow.new(
            mutation_id: mutation_id,
            sample_id: sample_id,
            cluster_id: cluster_id,
            cellular_prevalence: prevalence[mutation_offset + s_i],
            cellular_prevalence_std: prevalence_std[mutation_offset + s_i],
            cluster_assignment_prob: prob
          )
        end
      end

      rows.sort_by! { |row| {row.cluster_id, row.mutation_id, row.sample_id} }

      rows
    end
  end
end
