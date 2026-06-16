require "./spec_helper"

describe UnClone::Input do
  it "fills default tumour_content and error_rate" do
    path = File.join(Dir.tempdir, "unclone_input_#{Random.rand(1_000_000)}.tsv")
    begin
      File.write(
        path,
        "mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\tnormal_cn\n" +
        "m1\ts1\t10\t5\t2\t1\t2\n"
      )

      rows = UnClone::Input.read_tsv(path)
      rows.size.should eq(1)
      rows.first.tumour_content.should eq(1.0)
      rows.first.error_rate.should eq(0.001)
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "raises when required column is missing" do
    path = File.join(Dir.tempdir, "unclone_input_#{Random.rand(1_000_000)}.tsv")
    begin
      File.write(
        path,
        "mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\n" +
        "m1\ts1\t10\t5\t2\t1\n"
      )

      expect_raises(UnClone::CliError, /Missing required columns/) do
        UnClone::Input.read_tsv(path)
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "raises a line-numbered CliError for invalid numeric values" do
    path = File.join(Dir.tempdir, "unclone_input_#{Random.rand(1_000_000)}.tsv")
    begin
      File.write(
        path,
        "mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\tnormal_cn\n" +
        "m1\ts1\tnope\t5\t2\t1\t2\n"
      )

      expect_raises(UnClone::CliError, /Line 2: invalid integer for 'ref_counts': nope/) do
        UnClone::Input.read_tsv(path)
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "rejects invalid count and probability ranges" do
    path = File.join(Dir.tempdir, "unclone_input_#{Random.rand(1_000_000)}.tsv")
    begin
      File.write(
        path,
        "mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\tnormal_cn\ttumour_content\n" +
        "m1\ts1\t10\t5\t2\t1\t2\t1.5\n"
      )

      expect_raises(UnClone::CliError, /Line 2: tumour_content must be within \[0, 1\]/) do
        UnClone::Input.read_tsv(path)
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "accepts zero-depth rows as missing observations" do
    path = File.join(Dir.tempdir, "unclone_input_#{Random.rand(1_000_000)}.tsv")
    begin
      File.write(
        path,
        "mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\tnormal_cn\n" +
        "m1\ts1\t0\t0\t2\t1\t2\n"
      )

      rows = UnClone::Input.read_tsv(path)
      rows.size.should eq(1)
      rows.first.ref_counts.should eq(0)
      rows.first.alt_counts.should eq(0)
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end
