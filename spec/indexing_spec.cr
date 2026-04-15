require "./spec_helper"

private def make_row(mutation_id, sample_id, ref_counts, alt_counts, tumour_content = 0.7, error_rate = 0.001)
  Tyclone::InputRow.new(mutation_id, sample_id, ref_counts, alt_counts, 2, 1, 2, tumour_content, error_rate)
end

describe Tyclone::Indexing do
  describe ".build" do
    rows = [
      make_row("chrZ_99", "sample_b", 10, 5, 0.7),
      make_row("chrA_01", "sample_a", 20, 8, 0.8),
      make_row("chrZ_99", "sample_a", 15, 6, 0.7),
      make_row("chrA_01", "sample_b", 12, 4, 0.8),
    ]

    indexed = Tyclone::Indexing.build(rows)

    it "sorts mutation_ids alphabetically" do
      indexed.mutation_ids.should eq(["chrA_01", "chrZ_99"])
    end

    it "sorts sample_ids alphabetically" do
      indexed.sample_ids.should eq(["sample_a", "sample_b"])
    end

    it "assigns zero-based mutation indices" do
      indexed.mutation_to_index["chrA_01"].should eq(0)
      indexed.mutation_to_index["chrZ_99"].should eq(1)
    end

    it "assigns zero-based sample indices" do
      indexed.sample_to_index["sample_a"].should eq(0)
      indexed.sample_to_index["sample_b"].should eq(1)
    end

    it "returns correct num_mutations and num_samples" do
      indexed.num_mutations.should eq(2)
      indexed.num_samples.should eq(2)
    end

    it "copies FFI fields verbatim from InputRow" do
      # chrA_01/sample_a: mutation_index=0, sample_index=0, ref=20, alt=8, tc=0.8
      matches = indexed.rows.select { |r| r.mutation_index == 0 && r.sample_index == 0 }
      matches.size.should eq(1)
      ffi_row = matches.first
      ffi_row.ref_counts.should eq(20)
      ffi_row.alt_counts.should eq(8)
      ffi_row.tumour_content.should be_close(0.8, 1e-12)
      ffi_row.error_rate.should be_close(0.001, 1e-15)
    end

    it "sets correct mutation_index and sample_index in FFI rows" do
      # chrZ_99/sample_b: mutation_index=1, sample_index=1, ref=10, alt=5
      matches = indexed.rows.select { |r| r.mutation_index == 1 && r.sample_index == 1 }
      matches.size.should eq(1)
      ffi_row = matches.first
      ffi_row.ref_counts.should eq(10)
      ffi_row.alt_counts.should eq(5)
    end
  end
end
