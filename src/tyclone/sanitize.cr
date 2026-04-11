module Tyclone
  module Sanitize
    def self.run(rows : Array(InputRow)) : Array(InputRow)
      # Keep first exact duplicate only.
      deduped = rows.uniq do |row|
        {
          row.mutation_id,
          row.sample_id,
          row.ref_counts,
          row.alt_counts,
          row.major_cn,
          row.minor_cn,
          row.normal_cn,
          row.tumour_content,
          row.error_rate,
        }
      end

      no_zero_major = deduped.reject { |row| row.major_cn == 0 }
      unique_sample_ids = no_zero_major.map(&.sample_id)
      unique_sample_ids.uniq!
      sample_count = unique_sample_ids.size

      grouped = no_zero_major.group_by(&.mutation_id)
      valid_mutations = grouped.compact_map do |mutation_id, mutation_rows|
        sample_ids = mutation_rows.map(&.sample_id)
        next if sample_ids.uniq.size != sample_count
        next if sample_ids.size > sample_count
        mutation_id
      end.to_set

      no_zero_major.select { |row| valid_mutations.includes?(row.mutation_id) }
    end
  end
end
