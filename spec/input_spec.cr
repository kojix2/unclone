require "./spec_helper"

describe Tyclone::Input do
  it "fills default tumour_content and error_rate" do
    path = File.join(Dir.tempdir, "tyclone_input_#{Random.rand(1_000_000)}.tsv")
    begin
      File.write(
        path,
        "mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\tnormal_cn\n" +
        "m1\ts1\t10\t5\t2\t1\t2\n"
      )

      rows = Tyclone::Input.read_tsv(path)
      rows.size.should eq(1)
      rows.first.tumour_content.should eq(1.0)
      rows.first.error_rate.should eq(0.001)
    ensure
      File.delete(path) if File.exists?(path)
    end
  end

  it "raises when required column is missing" do
    path = File.join(Dir.tempdir, "tyclone_input_#{Random.rand(1_000_000)}.tsv")
    begin
      File.write(
        path,
        "mutation_id\tsample_id\tref_counts\talt_counts\tmajor_cn\tminor_cn\n" +
        "m1\ts1\t10\t5\t2\t1\n"
      )

      expect_raises(Tyclone::CliError, /Missing required columns/) do
        Tyclone::Input.read_tsv(path)
      end
    ensure
      File.delete(path) if File.exists?(path)
    end
  end
end
