module Tyclone
  struct IndexedRows
    getter rows : Array(LibPcv::PcvRow)
    getter mutation_ids : Array(String)
    getter sample_ids : Array(String)
    getter mutation_to_index : Hash(String, Int32)
    getter sample_to_index : Hash(String, Int32)

    def initialize(
      @rows : Array(LibPcv::PcvRow),
      @mutation_ids : Array(String),
      @sample_ids : Array(String),
      @mutation_to_index : Hash(String, Int32),
      @sample_to_index : Hash(String, Int32),
    )
    end

    def num_mutations : Int32
      @mutation_ids.size.to_i32
    end

    def num_samples : Int32
      @sample_ids.size.to_i32
    end
  end

  module Indexing
    def self.build(rows : Array(InputRow)) : IndexedRows
      mutation_ids = rows.map(&.mutation_id)
      mutation_ids.uniq!
      mutation_ids.sort!

      sample_ids = rows.map(&.sample_id)
      sample_ids.uniq!
      sample_ids.sort!

      mutation_to_index = {} of String => Int32
      mutation_ids.each_with_index { |id, i| mutation_to_index[id] = i.to_i32 }

      sample_to_index = {} of String => Int32
      sample_ids.each_with_index { |id, i| sample_to_index[id] = i.to_i32 }

      ffi_rows = rows.map do |row|
        LibPcv::PcvRow.new(
          mutation_index: mutation_to_index[row.mutation_id],
          sample_index: sample_to_index[row.sample_id],
          ref_counts: row.ref_counts,
          alt_counts: row.alt_counts,
          major_cn: row.major_cn,
          minor_cn: row.minor_cn,
          normal_cn: row.normal_cn,
          tumour_content: row.tumour_content,
          error_rate: row.error_rate
        )
      end

      IndexedRows.new(ffi_rows, mutation_ids, sample_ids, mutation_to_index, sample_to_index)
    end
  end
end
