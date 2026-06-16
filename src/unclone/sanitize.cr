module UnClone
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

      grouped = no_zero_major.group_by(&.mutation_id)
      valid_mutations = grouped.compact_map do |mutation_id, mutation_rows|
        sample_ids = mutation_rows.map(&.sample_id)
        next if sample_ids.uniq.size != sample_ids.size
        next unless mutation_rows.any? { |row| row.ref_counts + row.alt_counts > 0 }
        mutation_id
      end.to_set

      no_zero_major.select { |row| valid_mutations.includes?(row.mutation_id) }
    end
  end
end
