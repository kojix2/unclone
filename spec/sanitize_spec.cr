require "./spec_helper"

private def row(mutation_id, sample_id, major_cn = 2, ref_counts = 10)
  Tyclone::InputRow.new(
    mutation_id: mutation_id,
    sample_id: sample_id,
    ref_counts: ref_counts,
    alt_counts: 5,
    major_cn: major_cn,
    minor_cn: 1,
    normal_cn: 2,
    tumour_content: 1.0,
    error_rate: 0.001
  )
end

describe Tyclone::Sanitize do
  it "removes invalid mutations and keeps complete ones" do
    rows = [
      row("m1", "s1"),
      row("m1", "s2"),
      row("m2", "s1"),
      row("m3", "s1"),
      row("m3", "s1", 2, 11),
      row("m3", "s2"),
      row("m4", "s1", 0),
      row("m4", "s2"),
    ]

    sanitized = Tyclone::Sanitize.run(rows)
    mutation_ids = sanitized.map(&.mutation_id)
    mutation_ids.uniq!
    mutation_ids.should eq(["m1"])
    sanitized.size.should eq(2)
  end

  it "deduplicates exact duplicate rows" do
    rows = [
      row("m1", "s1"),
      row("m1", "s1"),
      row("m1", "s2"),
    ]

    sanitized = Tyclone::Sanitize.run(rows)
    sanitized.size.should eq(2)
    sanitized.count { |row| row.sample_id == "s1" }.should eq(1)
  end
end
