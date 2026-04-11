require "./spec_helper"
require "compress/gzip"

private def make_output_row
  Tyclone::OutputRow.new("mut1", "s1", 0, 0.5, 0.1, 0.9)
end

describe Tyclone::Output do
  describe ".write" do
    it "produces correct TSV header" do
      tmp = File.tempfile("tyclone_output_spec")
      path = tmp.path
      tmp.close
      begin
        Tyclone::Output.write(path, [] of Tyclone::OutputRow, false)
        content = File.read(path)
        content.should eq(
          "mutation_id\tsample_id\tcluster_id\tcellular_prevalence\tcellular_prevalence_std\tcluster_assignment_prob\n"
        )
      ensure
        File.delete(path) if File.exists?(path)
      end
    end

    it "writes data rows with correct field values" do
      tmp = File.tempfile("tyclone_output_spec")
      path = tmp.path
      tmp.close
      begin
        Tyclone::Output.write(path, [make_output_row], false)
        lines = File.read(path).lines
        lines.size.should eq(2)
        fields = lines[1].split('\t')
        fields[0].should eq("mut1")
        fields[1].should eq("s1")
        fields[2].should eq("0")
        fields[3].should eq("0.5")
        fields[4].should eq("0.1")
        fields[5].should eq("0.9")
      ensure
        File.delete(path) if File.exists?(path)
      end
    end

    it "empty rows produce header only" do
      tmp = File.tempfile("tyclone_output_spec")
      path = tmp.path
      tmp.close
      begin
        Tyclone::Output.write(path, [] of Tyclone::OutputRow, false)
        lines = File.read(path).lines
        lines.size.should eq(1)
      ensure
        File.delete(path) if File.exists?(path)
      end
    end

    it "gzip output is readable and contains expected content" do
      tmp = File.tempfile("tyclone_output_spec")
      path = tmp.path
      tmp.close
      begin
        Tyclone::Output.write(path, [make_output_row], true)
        content = File.open(path) do |file|
          Compress::Gzip::Reader.open(file, &.gets_to_end)
        end
        content.should contain("mutation_id\tsample_id\tcluster_id")
        content.should contain("mut1\ts1\t0")
      ensure
        File.delete(path) if File.exists?(path)
      end
    end
  end
end
