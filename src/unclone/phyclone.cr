require "csv"
require "json"
require "compress/gzip"

module UnClone
  module PhyClone
    TRACE_SCHEMA_VERSION = 1
    TRACE_ROOT_ID        = "root"
    TRACE_ROOT_KIND      = "root"

    struct TraceNode
      getter id : String
      getter kind : String
      getter cluster_ids : Array(Int32)

      def initialize(@id : String, @kind : String, @cluster_ids : Array(Int32))
      end
    end

    struct TraceEdge
      getter parent : String
      getter child : String

      def initialize(@parent : String, @child : String)
      end
    end

    struct TraceRecord
      getter chain : Int32
      getter iter : Int32
      getter log_p : Float64
      getter log_p_one : Float64
      getter topology_id : String
      getter nodes : Array(TraceNode)
      getter edges : Array(TraceEdge)
      getter clusters : Array(ClusterSummary)
      getter outlier_assignments : Array(OutlierAssignment)
      getter metadata : TraceMetadata?

      def initialize(
        @chain : Int32,
        @iter : Int32,
        @log_p : Float64,
        @log_p_one : Float64,
        @topology_id : String,
        @nodes : Array(TraceNode),
        @edges : Array(TraceEdge),
        @clusters : Array(ClusterSummary),
        @outlier_assignments : Array(OutlierAssignment) = [] of OutlierAssignment,
        @metadata : TraceMetadata? = nil,
      )
      end
    end

    struct OutlierAssignment
      getter mutation_id : String
      getter? outlier : Bool
      # nil when not computed by unclone (field is omitted from JSON output in that case)
      getter log_odds_outlier_vs_in_tree : Float64?

      def initialize(
        @mutation_id : String,
        @outlier : Bool,
        @log_odds_outlier_vs_in_tree : Float64?,
      )
      end
    end

    struct TraceMetadata
      getter num_chains : Int32
      getter num_iters : Int32
      getter seed : UInt64?

      def initialize(@num_chains : Int32, @num_iters : Int32, @seed : UInt64?)
      end
    end

    struct InputRow
      getter mutation_id : String
      getter sample_id : String
      getter ref_counts : Int32
      getter alt_counts : Int32
      getter major_cn : Int32
      getter minor_cn : Int32
      getter normal_cn : Int32
      getter tumour_content : Float64
      getter error_rate : Float64
      getter cluster_id : Int32?
      getter chrom : String?
      getter loss_prob : Float64?
      getter outlier_prob : Float64?

      def initialize(
        @mutation_id : String,
        @sample_id : String,
        @ref_counts : Int32,
        @alt_counts : Int32,
        @major_cn : Int32,
        @minor_cn : Int32,
        @normal_cn : Int32,
        @tumour_content : Float64,
        @error_rate : Float64,
        @cluster_id : Int32?,
        @chrom : String?,
        @loss_prob : Float64?,
        @outlier_prob : Float64?,
      )
      end
    end

    struct ClusterSummary
      getter cluster_id : Int32
      getter mutation_ids : Array(String)
      getter sample_ids : Array(String)
      getter sample_profiles : Array(SampleProfile)
      getter outlier_data_points : Array(OutlierDataPoint)

      def initialize(
        @cluster_id : Int32,
        @mutation_ids : Array(String),
        @sample_ids : Array(String),
        @sample_profiles : Array(SampleProfile) = [] of SampleProfile,
        @outlier_data_points : Array(OutlierDataPoint) = [] of OutlierDataPoint,
      )
      end
    end

    struct OutlierDataPoint
      getter name : String
      getter outlier_prob : Float64
      getter outlier_prob_not : Float64
      getter outlier_marginal_prob : Float64
      getter loss_log_prob : Float64
      getter sample_observations : Array(OutlierSampleObservation)

      def initialize(
        @name : String,
        @outlier_prob : Float64,
        @outlier_prob_not : Float64,
        @outlier_marginal_prob : Float64,
        @loss_log_prob : Float64 = 0.0,
        @sample_observations : Array(OutlierSampleObservation) = [] of OutlierSampleObservation,
      )
      end
    end

    struct OutlierSampleObservation
      getter sample_id : String
      getter ref_counts : Int32
      getter alt_counts : Int32
      getter major_cn : Int32
      getter minor_cn : Int32
      getter normal_cn : Int32
      getter tumour_content : Float64
      getter error_rate : Float64

      def initialize(
        @sample_id : String,
        @ref_counts : Int32,
        @alt_counts : Int32,
        @major_cn : Int32,
        @minor_cn : Int32,
        @normal_cn : Int32,
        @tumour_content : Float64,
        @error_rate : Float64,
      )
      end
    end

    struct SampleProfile
      getter sample_id : String
      getter ref_counts : Int32
      getter alt_counts : Int32

      def initialize(@sample_id : String, @ref_counts : Int32, @alt_counts : Int32)
      end
    end

    module Input
      REQUIRED_COLUMNS = {
        "mutation_id",
        "sample_id",
        "ref_counts",
        "alt_counts",
        "major_cn",
        "minor_cn",
        "normal_cn",
      }

      def self.read_tsv(path : String) : Array(InputRow)
        rows = [] of InputRow
        File.open(path) do |file|
          csv = CSV.new(file, headers: true, separator: '\t')
          headers = csv.headers || [] of String
          missing = REQUIRED_COLUMNS.reject { |column| headers.includes?(column) }
          unless missing.empty?
            raise CliError.new("Missing required columns for phy run: #{missing.join(", ")}")
          end

          line_number = 1
          csv.each do |row|
            line_number += 1
            cluster_id = optional_i32(row, "cluster_id", line_number)
            chrom = row["chrom"]?.try(&.presence)
            outlier_prob = optional_probability(row, "outlier_prob", line_number)
            loss_prob = optional_probability(row, "loss_prob", line_number)
            ref_counts = parse_i32(row, "ref_counts", line_number)
            alt_counts = parse_i32(row, "alt_counts", line_number)
            major_cn = parse_i32(row, "major_cn", line_number)
            minor_cn = parse_i32(row, "minor_cn", line_number)
            normal_cn = parse_i32(row, "normal_cn", line_number)
            tumour_content = parse_f64(row, "tumour_content", line_number, "1.0")
            error_rate = parse_f64(row, "error_rate", line_number, "0.001")

            validate_values(
              line_number,
              ref_counts,
              alt_counts,
              major_cn,
              minor_cn,
              normal_cn,
              tumour_content,
              error_rate
            )

            rows << InputRow.new(
              mutation_id: required(row, "mutation_id", line_number),
              sample_id: required(row, "sample_id", line_number),
              ref_counts: ref_counts,
              alt_counts: alt_counts,
              major_cn: major_cn,
              minor_cn: minor_cn,
              normal_cn: normal_cn,
              tumour_content: tumour_content,
              error_rate: error_rate,
              cluster_id: cluster_id,
              chrom: chrom,
              loss_prob: loss_prob,
              outlier_prob: outlier_prob,
            )
          end
        end
        rows
      end

      # Extended cluster file row: includes optional cellular_prevalence and outlier_prob
      struct ClusterRow
        getter mutation_id : String
        getter cluster_id : String
        getter sample_id : String?
        getter chrom : String?
        getter cellular_prevalence : Float64?
        getter outlier_prob : Float64?

        def initialize(
          @mutation_id : String,
          @cluster_id : String,
          @sample_id : String?,
          @chrom : String?,
          @cellular_prevalence : Float64?,
          @outlier_prob : Float64?,
        )
        end
      end

      def self.read_cluster_tsv(path : String) : Hash(String, Int32)
        read_cluster_tsv_full(path).each_with_object({} of String => Int32) do |row, mapping|
          cluster_id = row.cluster_id.to_i32? || raise CliError.new("cluster-file has invalid integer cluster_id for mutation '#{row.mutation_id}': #{row.cluster_id}")
          if mapping.has_key?(row.mutation_id) && mapping[row.mutation_id] != cluster_id
            raise CliError.new("cluster-file has conflicting cluster_id for mutation '#{row.mutation_id}'")
          end
          mapping[row.mutation_id] = cluster_id
        end
      end

      def self.read_cluster_tsv_full(path : String) : Array(ClusterRow)
        rows = [] of ClusterRow
        File.open(path) do |file|
          csv = CSV.new(file, headers: true, separator: '\t')
          headers = csv.headers || [] of String
          unless headers.includes?("mutation_id") && headers.includes?("cluster_id")
            raise CliError.new("cluster-file must include mutation_id and cluster_id columns")
          end

          line_number = 1
          csv.each do |row|
            line_number += 1
            mutation_id = required(row, "mutation_id", line_number)
            cluster_id = required(row, "cluster_id", line_number)
            sample_id = headers.includes?("sample_id") ? row["sample_id"]?.try(&.presence) : nil
            chrom = headers.includes?("chrom") ? row["chrom"]?.try(&.presence) : nil
            cellular_prevalence = headers.includes?("cellular_prevalence") ? optional_probability(row, "cellular_prevalence", line_number) : nil
            outlier_prob = headers.includes?("outlier_prob") ? optional_probability(row, "outlier_prob", line_number) : nil
            rows << ClusterRow.new(mutation_id, cluster_id, sample_id, chrom, cellular_prevalence, outlier_prob)
          end
        end
        rows
      end

      private def self.required(row : CSV, key : String, line_number : Int32) : String
        value = row[key]?
        if value.nil? || value.empty?
          raise CliError.new("Line #{line_number}: missing value for '#{key}' in phy input")
        end
        value
      end

      private def self.optional(row : CSV, key : String, default_value : String) : String
        value = row[key]?
        return default_value if value.nil? || value.empty?
        value
      end

      private def self.parse_i32(row : CSV, key : String, line_number : Int32) : Int32
        value = required(row, key, line_number)
        value.to_i32? || raise CliError.new("Line #{line_number}: invalid integer for '#{key}': #{value}")
      end

      private def self.optional_i32(row : CSV, key : String, line_number : Int32) : Int32?
        value = row[key]?.try(&.presence)
        return nil unless value
        value.to_i32? || raise CliError.new("Line #{line_number}: invalid integer for '#{key}': #{value}")
      end

      private def self.parse_f64(row : CSV, key : String, line_number : Int32, default_value : String) : Float64
        value = optional(row, key, default_value)
        parsed = value.to_f64?
        unless parsed && parsed.finite?
          raise CliError.new("Line #{line_number}: invalid number for '#{key}': #{value}")
        end
        parsed
      end

      private def self.optional_probability(row : CSV, key : String, line_number : Int32) : Float64?
        value = row[key]?.try(&.presence)
        return nil unless value
        parsed = value.to_f64?
        unless parsed && parsed.finite?
          raise CliError.new("Line #{line_number}: invalid number for '#{key}': #{value}")
        end
        unless (0.0..1.0).includes?(parsed)
          raise CliError.new("Line #{line_number}: #{key} must be within [0, 1]")
        end
        parsed
      end

      private def self.validate_values(
        line_number : Int32,
        ref_counts : Int32,
        alt_counts : Int32,
        major_cn : Int32,
        minor_cn : Int32,
        normal_cn : Int32,
        tumour_content : Float64,
        error_rate : Float64,
      ) : Nil
        raise CliError.new("Line #{line_number}: ref_counts must be >= 0") if ref_counts < 0
        raise CliError.new("Line #{line_number}: alt_counts must be >= 0") if alt_counts < 0
        raise CliError.new("Line #{line_number}: total read depth must be > 0") if ref_counts + alt_counts <= 0
        raise CliError.new("Line #{line_number}: major_cn must be >= 0") if major_cn < 0
        raise CliError.new("Line #{line_number}: minor_cn must be >= 0") if minor_cn < 0
        raise CliError.new("Line #{line_number}: normal_cn must be >= 0") if normal_cn < 0
        raise CliError.new("Line #{line_number}: minor_cn must be <= major_cn") if minor_cn > major_cn
        raise CliError.new("Line #{line_number}: tumour_content must be within [0, 1]") unless (0.0..1.0).includes?(tumour_content)
        raise CliError.new("Line #{line_number}: error_rate must be within [0, 1]") unless (0.0..1.0).includes?(error_rate)
      end
    end

    def self.run(config : PhyCloneRunConfig)
      rows = Input.read_tsv(config.in_file)
      raise CliError.new("No rows found in phy input") if rows.empty?
      raise CliError.new("--num-iters must be > 0") if config.num_iters <= 0
      raise CliError.new("--num-chains must be > 0") if config.num_chains <= 0

      cluster_rows = config.cluster_file.try { |path| Input.read_cluster_tsv_full(path) }

      # Validate that cluster-file mutations exist in the input TSV.
      if cluster_rows
        input_mutations = Set(String).new(rows.map(&.mutation_id))
        cluster_mutations = cluster_rows.map(&.mutation_id)
        cluster_mutations.uniq!
        unknown = cluster_mutations.reject { |mid| input_mutations.includes?(mid) }
        unless unknown.empty?
          raise CliError.new("cluster-file contains mutations not found in input: #{unknown.sort.join(", ")}")
        end
      end

      write_trace(
        config.out_file,
        config.compress?,
        PhyCloneKernel.generate_trace(rows, cluster_rows, config, config.num_chains, config.num_iters, config.seed)
      )
    end

    def self.map(config : PhyCloneMapConfig)
      summary = map_trace_summary(config.in_file)
      raise CliError.new("No trace records found in phy trace") unless summary

      write_trace_summary(config.out_file, summary)
    end

    def self.consensus(config : PhyCloneConsensusConfig)
      summary = consensus_trace_summary(config.in_file, config.consensus_threshold, config.weight)
      raise CliError.new("No trace records found in phy trace") unless summary

      # Build clade-based consensus alongside topology-representative consensus
      clade_support, clade_total = build_clade_support(config.in_file)
      threshold_count = (config.consensus_threshold * clade_total).ceil.to_i
      consensus_clades = clade_support.keys.select { |k| clade_support[k] >= threshold_count }
      consensus_clades = clade_support.keys if consensus_clades.empty?
      clade_parent_map = build_clade_parent_map(consensus_clades)
      consensus_tree = build_consensus_tree(consensus_clades, clade_parent_map)

      File.open(config.out_file, "w") do |file|
        JSON.build(file) do |json|
          json.object do
            json.field "schema_version", TRACE_SCHEMA_VERSION
            json.field "topology_id", summary.topology_id
            json.field "newick", to_newick(summary.record)
            json.field "support", summary.support
            json.field "num_records", summary.num_records
            json.field "support_fraction", summary.support.to_f64 / summary.num_records.to_f64
            json.field "consensus_threshold", config.consensus_threshold
            json.field "weight", config.weight
            json.field "representative" do
              json.object do
                json.field "chain", summary.record.chain
                json.field "iter", summary.record.iter
                json.field "log_p", summary.record.log_p
                json.field "log_p_one", summary.record.log_p_one
              end
            end
            json.field "clusters" do
              json.array do
                summary.record.clusters.each do |cluster|
                  json.object do
                    json.field "cluster_id", cluster.cluster_id
                    json.field "mutation_ids", cluster.mutation_ids
                    json.field "sample_ids", cluster.sample_ids
                  end
                end
              end
            end
            json.field "clade_consensus" do
              json.object do
                json.field "num_clades", consensus_clades.size
                json.field "clades" do
                  json.array do
                    consensus_clades.sort.each do |clade_k|
                      json.object do
                        json.field "clade", clade_k.split("|")
                        json.field "exclusive_ids", exclusive_ids(clade_k, clade_parent_map)
                        json.field "support", clade_support[clade_k]
                        json.field "support_fraction", clade_support[clade_k].to_f64 / clade_total.to_f64
                        json.field "parent_clade", clade_parent_map[clade_k]?.try(&.split("|")) || [] of String
                      end
                    end
                  end
                end
              end
            end
            json.field "consensus_tree" do
              json.object do
                json.field "nodes" do
                  json.array do
                    consensus_tree[:nodes].each do |node|
                      json.object do
                        json.field "id", node[:id]
                        json.field "clade", node[:clade]
                        json.field "exclusive_ids", node[:exclusive_ids]
                      end
                    end
                  end
                end
                json.field "edges" do
                  json.array do
                    consensus_tree[:edges].each do |edge|
                      json.object do
                        json.field "parent", edge[:parent]
                        json.field "child", edge[:child]
                      end
                    end
                  end
                end
                json.field "newick", consensus_tree[:newick]
              end
            end
          end
        end
      end
    end

    private struct ConsensusTraceSummary
      getter topology_id : String
      getter support : Int32
      getter num_records : Int32
      getter record : TraceRecord

      def initialize(@topology_id : String, @support : Int32, @num_records : Int32, @record : TraceRecord)
      end
    end

    private struct TopologySummary
      getter topology_id : String
      getter support : Int32
      getter record : TraceRecord

      def initialize(@topology_id : String, @support : Int32, @record : TraceRecord)
      end
    end

    private struct TraceScanResult
      getter num_records : Int32
      getter best_record : TraceRecord?
      getter support_by_topology : Hash(String, Int32)
      getter representative_by_topology : Hash(String, TraceRecord)
      getter trace_metadata : TraceMetadata?

      def initialize(
        @num_records : Int32,
        @best_record : TraceRecord?,
        @support_by_topology : Hash(String, Int32),
        @representative_by_topology : Hash(String, TraceRecord),
        @trace_metadata : TraceMetadata?,
      )
      end
    end

    private struct MapTraceSummary
      getter record : TraceRecord
      getter num_records : Int32
      getter cluster_prevalence_ci : Hash(Int32, Hash(String, {Float64, Float64, Float64}))
      getter clonal_prevalence_ci : Hash(Int32, Hash(String, {Float64, Float64, Float64}))

      def initialize(
        @record : TraceRecord,
        @num_records : Int32,
        @cluster_prevalence_ci : Hash(Int32, Hash(String, {Float64, Float64, Float64})),
        @clonal_prevalence_ci : Hash(Int32, Hash(String, {Float64, Float64, Float64})),
      )
      end
    end

    private def self.write_trace_summary(path : String, summary : MapTraceSummary)
      record = summary.record
      cluster_prevalence = compute_cluster_sample_prevalence(record)
      clonal_prevalence = compute_clonal_prevalence(record, cluster_prevalence)

      File.open(path, "w") do |file|
        JSON.build(file) do |json|
          json.object do
            json.field "schema_version", TRACE_SCHEMA_VERSION
            json.field "chain", record.chain
            json.field "iter", record.iter
            json.field "log_p", record.log_p
            json.field "log_p_one", record.log_p_one
            json.field "topology_id", record.topology_id
            json.field "newick", to_newick(record)
            json.field "num_records", summary.num_records
            json.field "map_method", "best_record_with_trace_quantiles"
            json.field "clusters" do
              json.array do
                record.clusters.each do |cluster|
                  json.object do
                    json.field "cluster_id", cluster.cluster_id
                    json.field "mutation_ids", cluster.mutation_ids
                    json.field "sample_ids", cluster.sample_ids
                  end
                end
              end
            end
            json.field "outlier_assignments" do
              json.array do
                record.outlier_assignments.each do |assignment|
                  json.object do
                    json.field "mutation_id", assignment.mutation_id
                    json.field "is_outlier", assignment.outlier?
                    # Omit log_odds_outlier_vs_in_tree when nil (not computed by unclone)
                    if log_odds = assignment.log_odds_outlier_vs_in_tree
                      json.field "log_odds_outlier_vs_in_tree", log_odds
                    end
                  end
                end
              end
            end
            json.field "outlier_summary" do
              json.object do
                num_outliers = record.outlier_assignments.count(&.outlier?)
                json.field "num_assignments", record.outlier_assignments.size
                json.field "num_outliers", num_outliers
                json.field "num_in_tree", record.outlier_assignments.size - num_outliers
              end
            end
            json.field "cluster_sample_prevalence" do
              json.object do
                cluster_ids = cluster_prevalence.keys
                cluster_ids.sort!
                cluster_ids.each do |cluster_id|
                  json.field cluster_id.to_s do
                    json.object do
                      samples = cluster_prevalence[cluster_id]
                      sample_ids = samples.keys
                      sample_ids.sort!
                      sample_ids.each do |sample_id|
                        json.field sample_id, samples[sample_id]
                      end
                    end
                  end
                end
              end
            end
            json.field "clonal_prevalence" do
              json.object do
                cluster_ids = clonal_prevalence.keys
                cluster_ids.sort!
                cluster_ids.each do |cluster_id|
                  json.field cluster_id.to_s do
                    json.object do
                      samples = clonal_prevalence[cluster_id]
                      sample_ids = samples.keys
                      sample_ids.sort!
                      sample_ids.each do |sample_id|
                        json.field sample_id, samples[sample_id]
                      end
                    end
                  end
                end
              end
            end
            json.field "cluster_sample_prevalence_ci" do
              json.object do
                cluster_ids = summary.cluster_prevalence_ci.keys
                cluster_ids.sort!
                cluster_ids.each do |cluster_id|
                  json.field cluster_id.to_s do
                    json.object do
                      samples = summary.cluster_prevalence_ci[cluster_id]
                      sample_ids = samples.keys
                      sample_ids.sort!
                      sample_ids.each do |sample_id|
                        lower, median, upper = samples[sample_id]
                        json.field sample_id do
                          json.object do
                            json.field "lower", lower
                            json.field "median", median
                            json.field "upper", upper
                          end
                        end
                      end
                    end
                  end
                end
              end
            end
            json.field "clonal_prevalence_ci" do
              json.object do
                cluster_ids = summary.clonal_prevalence_ci.keys
                cluster_ids.sort!
                cluster_ids.each do |cluster_id|
                  json.field cluster_id.to_s do
                    json.object do
                      samples = summary.clonal_prevalence_ci[cluster_id]
                      sample_ids = samples.keys
                      sample_ids.sort!
                      sample_ids.each do |sample_id|
                        lower, median, upper = samples[sample_id]
                        json.field sample_id do
                          json.object do
                            json.field "lower", lower
                            json.field "median", median
                            json.field "upper", upper
                          end
                        end
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end
    end

    private def self.map_trace_summary(path : String) : MapTraceSummary?
      best_record = nil.as(TraceRecord?)
      num_records = 0
      cluster_samples = Hash(Int32, Hash(String, Array(Float64))).new { |hash, key| hash[key] = Hash(String, Array(Float64)).new { |inner, sample_id| inner[sample_id] = [] of Float64 } }
      clonal_samples = Hash(Int32, Hash(String, Array(Float64))).new { |hash, key| hash[key] = Hash(String, Array(Float64)).new { |inner, sample_id| inner[sample_id] = [] of Float64 } }

      each_trace_record(path) do |record|
        num_records += 1
        current_best = best_record
        if current_best.nil? || record.log_p > current_best.log_p
          best_record = record
        end

        cluster_prevalence = compute_cluster_sample_prevalence(record)
        clonal_prevalence = compute_clonal_prevalence(record, cluster_prevalence)

        cluster_prevalence.each do |cluster_id, by_sample|
          by_sample.each do |sample_id, value|
            cluster_samples[cluster_id][sample_id] << value
          end
        end

        clonal_prevalence.each do |cluster_id, by_sample|
          by_sample.each do |sample_id, value|
            clonal_samples[cluster_id][sample_id] << value
          end
        end
      end

      record = best_record
      return if num_records == 0 || record.nil?

      cluster_prevalence_ci = summarize_prevalence_samples(cluster_samples)
      clonal_prevalence_ci = summarize_prevalence_samples(clonal_samples)

      MapTraceSummary.new(record, num_records, cluster_prevalence_ci, clonal_prevalence_ci)
    end

    private def self.summarize_prevalence_samples(
      samples : Hash(Int32, Hash(String, Array(Float64))),
    ) : Hash(Int32, Hash(String, {Float64, Float64, Float64}))
      result = {} of Int32 => Hash(String, {Float64, Float64, Float64})

      samples.each do |cluster_id, by_sample|
        summary_by_sample = {} of String => {Float64, Float64, Float64}
        by_sample.each do |sample_id, values|
          sorted = values.sort
          lower = percentile(sorted, 0.05)
          median = percentile(sorted, 0.5)
          upper = percentile(sorted, 0.95)
          summary_by_sample[sample_id] = {lower, median, upper}
        end
        result[cluster_id] = summary_by_sample
      end

      result
    end

    private def self.percentile(sorted_values : Array(Float64), q : Float64) : Float64
      return 0.0 if sorted_values.empty?
      return sorted_values.first if sorted_values.size == 1

      clamped_q = q.clamp(0.0, 1.0)
      pos = clamped_q * (sorted_values.size - 1)
      low_idx = pos.floor.to_i
      high_idx = pos.ceil.to_i
      return sorted_values[low_idx] if low_idx == high_idx

      low = sorted_values[low_idx]
      high = sorted_values[high_idx]
      low + (high - low) * (pos - low_idx)
    end

    private def self.compute_cluster_sample_prevalence(record : TraceRecord) : Hash(Int32, Hash(String, Float64))
      prevalence = {} of Int32 => Hash(String, Float64)

      record.clusters.each do |cluster|
        by_sample = {} of String => Float64
        cluster.sample_profiles.each do |profile|
          total = profile.ref_counts + profile.alt_counts
          value = if total <= 0
                    0.0
                  else
                    ((2.0 * profile.alt_counts.to_f64) / total.to_f64).clamp(0.0, 1.0)
                  end
          by_sample[profile.sample_id] = value
        end
        prevalence[cluster.cluster_id] = by_sample
      end

      prevalence
    end

    private def self.compute_clonal_prevalence(
      record : TraceRecord,
      cluster_prevalence : Hash(Int32, Hash(String, Float64)),
    ) : Hash(Int32, Hash(String, Float64))
      # Map each node to its first cluster_id (for edge traversal).
      # Nodes with multiple cluster_ids are multi-cluster nodes; use the first id for tree structure.
      node_to_cluster = {} of String => Int32
      record.nodes.each do |node|
        next unless node.kind == "cluster"
        cluster_id = node.cluster_ids.first?
        node_to_cluster[node.id] = cluster_id if cluster_id
      end

      children_by_parent = Hash(Int32, Array(Int32)).new { |hash, key| hash[key] = [] of Int32 }
      record.edges.each do |edge|
        parent_cluster = node_to_cluster[edge.parent]?
        child_cluster = node_to_cluster[edge.child]?
        next unless parent_cluster && child_cluster
        children_by_parent[parent_cluster] << child_cluster
      end

      result = {} of Int32 => Hash(String, Float64)
      cluster_prevalence.each do |cluster_id, by_sample|
        clonal_by_sample = {} of String => Float64
        children = children_by_parent[cluster_id]? || [] of Int32

        by_sample.each do |sample_id, parent_value|
          children_sum = children.sum(0.0) do |child_id|
            cluster_prevalence[child_id]?.try(&.[sample_id]?) || 0.0
          end
          clonal_by_sample[sample_id] = (parent_value - children_sum).clamp(0.0, 1.0)
        end

        result[cluster_id] = clonal_by_sample
      end

      result
    end

    # Encode a set of cluster ids as a canonical string key for use in Hash/Set.
    private def self.clade_key(cluster_ids : Array(String)) : String
      cluster_ids.sort.join("|")
    end

    # Collect cluster-node ids reachable from *root_id* (inclusive) via directed edges.
    private def self.collect_descendants(root_id : String, children_map : Hash(String, Array(String))) : Array(String)
      result = [] of String
      stack = [root_id]
      until stack.empty?
        node = stack.pop
        result << node
        (children_map[node]? || [] of String).each { |child| stack << child }
      end
      result
    end

    # Return one clade key per cluster node in the record.
    # A clade is defined as the set of *cluster_ids* (biological mutation IDs) contained
    # in the subtree rooted at that node.  Node IDs (e.g. "shell-2") are internal and
    # may differ between MCMC runs for the same biological topology, so we expand each
    # descendant node's cluster_ids instead of using node IDs directly.
    private def self.extract_clades(record : TraceRecord) : Array(String)
      cluster_nodes = record.nodes.select { |node| node.kind == "cluster" }
      return [] of String if cluster_nodes.empty?

      children_map = Hash(String, Array(String)).new { |hash, key| hash[key] = [] of String }
      record.edges.each do |edge|
        children_map[edge.parent] << edge.child
      end

      node_by_id = record.nodes.to_h { |node| {node.id, node} }

      cluster_nodes.map do |node|
        descendant_ids = collect_descendants(node.id, children_map)
        cluster_ids = descendant_ids.flat_map do |nid|
          node_by_id[nid]?.try(&.cluster_ids) || [] of Int32
        end
        cluster_ids.uniq!
        cluster_ids.sort!
        clade_key(cluster_ids.map(&.to_s))
      end
    end

    # Scan trace and accumulate per-clade support counts.
    # Returns {clade_key => count, total_records}.
    private def self.build_clade_support(path : String) : {Hash(String, Int32), Int32}
      support = Hash(String, Int32).new(0)
      total = 0
      each_trace_record(path) do |record|
        total += 1
        extract_clades(record).each { |key| support[key] += 1 }
      end
      {support, total}
    end

    # Given a set of canonical clade keys, find the smallest strict superset of *query*.
    # Returns nil if no superset is found.
    private def self.find_parent_clade(all_clades : Array(String), query : String) : String?
      query_ids = query.split("|")
      best_parent : String? = nil
      best_parent_size = Int32::MAX
      all_clades.each do |candidate|
        next if candidate == query
        candidate_ids = candidate.split("|")
        next if candidate_ids.size >= best_parent_size
        next unless query_ids.all? { |id| candidate_ids.includes?(id) }
        best_parent = candidate
        best_parent_size = candidate_ids.size
      end
      best_parent
    end

    # Build a parent map for a set of clades using smallest-superset parentage.
    # Returns Hash(clade_key => parent_clade_key | nil-for-roots).
    private def self.build_clade_parent_map(clades : Array(String)) : Hash(String, String?)
      parent_map = Hash(String, String?).new
      clades.each { |clade| parent_map[clade] = find_parent_clade(clades, clade) }
      parent_map
    end

    # Compute per-clade exclusive ids (clade minus union of children clades).
    private def self.exclusive_ids(clade_key : String, parent_map : Hash(String, String?)) : Array(String)
      own_ids = clade_key.split("|")
      children_ids = parent_map.select { |_, parent| parent == clade_key }.keys.flat_map(&.split("|"))
      own_ids.reject { |id| children_ids.includes?(id) }
    end

    private def self.build_consensus_tree(
      clades : Array(String),
      parent_map : Hash(String, String?),
    ) : NamedTuple(
      nodes: Array(NamedTuple(id: String, clade: Array(String), exclusive_ids: Array(String))),
      edges: Array(NamedTuple(parent: String, child: String)),
      newick: String)
      sorted = clades.sort
      id_by_clade = {} of String => String
      sorted.each_with_index do |clade, i|
        id_by_clade[clade] = "clade-#{i}"
      end

      nodes = sorted.map do |clade|
        {
          id:            id_by_clade[clade],
          clade:         clade.split("|"),
          exclusive_ids: exclusive_ids(clade, parent_map),
        }
      end

      edges = [] of NamedTuple(parent: String, child: String)
      parent_map.each do |clade, parent|
        next unless parent
        parent_id = id_by_clade[parent]?
        child_id = id_by_clade[clade]?
        next unless parent_id && child_id
        edges << {parent: parent_id, child: child_id}
      end

      children_by_parent = Hash(String, Array(String)).new { |hash, key| hash[key] = [] of String }
      parent_map.each do |clade, parent|
        next unless parent
        children_by_parent[parent] << clade
      end

      roots = sorted.select { |clade| parent_map[clade]?.nil? }
      roots = [sorted.max_by(&.split("|").size)] if roots.empty? && !sorted.empty?

      root_parts = roots.compact_map do |root|
        build_consensus_newick_for_clade(root, children_by_parent, parent_map)
      end
      newick = if root_parts.empty?
                 ";"
               elsif root_parts.size == 1
                 "#{root_parts[0]};"
               else
                 "(#{root_parts.join(",")});"
               end

      {nodes: nodes, edges: edges, newick: newick}
    end

    private def self.build_consensus_newick_for_clade(
      clade : String,
      children_by_parent : Hash(String, Array(String)),
      parent_map : Hash(String, String?),
    ) : String
      children = children_by_parent[clade]? || [] of String
      children_parts = children.map { |child| build_consensus_newick_for_clade(child, children_by_parent, parent_map) }
      own_parts = exclusive_ids(clade, parent_map).map(&.gsub("-", "_"))
      parts = children_parts + own_parts

      if parts.empty?
        "node"
      elsif parts.size == 1
        parts[0]
      else
        "(#{parts.join(",")})"
      end
    end

    private def self.consensus_trace_summary(path : String, threshold : Float64, weight : String) : ConsensusTraceSummary?
      scan = scan_trace(path)
      return if scan.num_records == 0

      eligible_topology_ids = scan.support_by_topology.keys.select do |topology_id|
        support = scan.support_by_topology[topology_id]
        (support.to_f64 / scan.num_records.to_f64) >= threshold
      end
      eligible_topology_ids = scan.support_by_topology.keys if eligible_topology_ids.empty?

      best_topology_id = eligible_topology_ids.max_by do |topology_id|
        representative = scan.representative_by_topology[topology_id]
        if weight == "log_p"
          {representative.log_p, scan.support_by_topology[topology_id], -representative.chain, -representative.iter}
        else
          {scan.support_by_topology[topology_id], representative.log_p, -representative.chain, -representative.iter}
        end
      end
      representative = scan.representative_by_topology[best_topology_id]
      ConsensusTraceSummary.new(best_topology_id, scan.support_by_topology[best_topology_id], scan.num_records, representative)
    end

    def self.topology_report(config : PhyCloneTopologyReportConfig)
      scan = scan_trace(config.in_file)
      summaries = build_topology_summaries(scan)
      num_records = scan.num_records
      raise CliError.new("No trace records found in phy trace") if num_records == 0

      File.open(config.out_file, "w") do |file|
        JSON.build(file) do |json|
          json.object do
            json.field "schema_version", TRACE_SCHEMA_VERSION
            json.field "num_records", num_records
            json.field "num_topologies", summaries.size
            if metadata = scan.trace_metadata
              json.field "trace_metadata" do
                json.object do
                  json.field "num_chains", metadata.num_chains
                  json.field "num_iters", metadata.num_iters
                  json.field "seed", metadata.seed
                end
              end
            end
            json.field "topologies" do
              json.array do
                summaries.each do |summary|
                  json.object do
                    json.field "topology_id", summary.topology_id
                    json.field "support", summary.support
                    json.field "support_fraction", summary.support.to_f64 / num_records.to_f64
                    json.field "best_log_p", summary.record.log_p
                    json.field "newick", to_newick(summary.record)
                    json.field "representative" do
                      json.object do
                        json.field "chain", summary.record.chain
                        json.field "iter", summary.record.iter
                      end
                    end
                  end
                end
              end
            end
          end
        end
      end
    end

    private def self.build_topology_summaries(scan : TraceScanResult) : Array(TopologySummary)
      topology_ids = scan.support_by_topology.keys
      topology_ids.sort_by! do |topology_id|
        representative = scan.representative_by_topology[topology_id]
        {-scan.support_by_topology[topology_id], -representative.log_p, representative.chain, representative.iter, topology_id}
      end

      topology_ids.map do |topology_id|
        TopologySummary.new(topology_id, scan.support_by_topology[topology_id], scan.representative_by_topology[topology_id])
      end
    end

    private def self.scan_trace(path : String) : TraceScanResult
      support_by_topology = {} of String => Int32
      representative_by_topology = {} of String => TraceRecord
      num_records = 0
      best_record = nil.as(TraceRecord?)
      trace_metadata = nil.as(TraceMetadata?)

      each_trace_record(path) do |record|
        num_records += 1

        current_best = best_record
        if current_best.nil? || record.log_p > current_best.log_p
          best_record = record
        end

        topology_id = record.topology_id
        support_by_topology[topology_id] = support_by_topology.fetch(topology_id, 0) + 1
        trace_metadata ||= record.metadata

        current_representative = representative_by_topology[topology_id]?
        if current_representative.nil? || record.log_p > current_representative.log_p
          representative_by_topology[topology_id] = record
        end
      end

      TraceScanResult.new(num_records, best_record, support_by_topology, representative_by_topology, trace_metadata)
    end

    private def self.build_clusters(
      rows : Array(InputRow),
      config : PhyCloneRunConfig,
      cluster_mapping : Hash(String, Int32)?,
      loss_prob_override : Hash(String, Float64) = {} of String => Float64,
    ) : Array(ClusterSummary)
      next_cluster_id = 0
      mutation_cluster_ids = {} of String => Int32
      grouped = {} of Int32 => Array(InputRow)

      if cluster_mapping
        input_mutations = Set(String).new(rows.map(&.mutation_id))
        unknown = cluster_mapping.keys.reject { |mutation_id| input_mutations.includes?(mutation_id) }
        unless unknown.empty?
          raise CliError.new("cluster-file contains mutations not found in input: #{unknown.sort.join(", ")}")
        end
      end

      rows.each do |row|
        cluster_id = row.cluster_id
        mapped_cluster_id = cluster_mapping.try(&.[row.mutation_id]?)

        if cluster_id && mapped_cluster_id && cluster_id != mapped_cluster_id
          raise CliError.new("cluster_id mismatch for mutation '#{row.mutation_id}' between input and cluster-file")
        end

        cluster_id ||= mapped_cluster_id
        if cluster_id.nil?
          cluster_id = mutation_cluster_ids[row.mutation_id]?
          unless cluster_id
            cluster_id = next_cluster_id
            mutation_cluster_ids[row.mutation_id] = cluster_id
            next_cluster_id += 1
          end
        end

        bucket = grouped[cluster_id]?
        if bucket
          bucket << row
        else
          grouped[cluster_id] = [row]
        end
      end

      cluster_ids = grouped.keys
      cluster_ids.sort!
      cluster_ids.map do |cluster_id|
        cluster_rows = grouped[cluster_id]
        mutation_ids = cluster_rows.map(&.mutation_id)
        mutation_ids.uniq!
        mutation_ids.sort!
        sample_ids = cluster_rows.map(&.sample_id)
        sample_ids.uniq!
        sample_ids.sort!
        sample_profiles = build_sample_profiles(cluster_rows)
        outlier_data_points = build_outlier_data_points(cluster_rows, config, loss_prob_override)
        ClusterSummary.new(
          cluster_id,
          mutation_ids,
          sample_ids,
          sample_profiles,
          outlier_data_points
        )
      end
    end

    private def self.build_outlier_data_points(
      rows : Array(InputRow),
      config : PhyCloneRunConfig,
      loss_prob_override : Hash(String, Float64) = {} of String => Float64,
    ) : Array(OutlierDataPoint)
      grouped = rows.group_by(&.mutation_id)
      mutation_ids = grouped.keys
      mutation_ids.sort!

      mutation_ids.map do |mutation_id|
        mutation_rows = grouped[mutation_id]
        avg_error_rate = mutation_rows.sum(&.error_rate) / mutation_rows.size
        p_outlier = avg_error_rate.clamp(1e-9, 1.0 - 1e-9)
        p_not = 1.0 - p_outlier
        loss_log_prob = loss_prob_override[mutation_id]?.try { |prob| Math.log(prob.clamp(1e-9, 1.0 - 1e-9)) } ||
                        compute_loss_log_prob(mutation_rows, config)
        sample_observations = mutation_rows
          .sort_by(&.sample_id)
          .map do |row|
            OutlierSampleObservation.new(
              row.sample_id,
              row.ref_counts,
              row.alt_counts,
              row.major_cn,
              row.minor_cn,
              row.normal_cn,
              row.tumour_content,
              row.error_rate
            )
          end

        OutlierDataPoint.new(
          mutation_id,
          Math.log(p_outlier),
          Math.log(p_not),
          Math.log((0.5 * p_outlier) + (0.5 * p_not)),
          loss_log_prob,
          sample_observations
        )
      end
    end

    # Derive per-mutation loss probability from cluster file cellular_prevalence.
    # Mutations whose mean cellular_prevalence is low (< loss_prob threshold) use high_loss_prob.
    private def self.derive_loss_probs_from_prevalence(
      cluster_rows : Hash(String, Input::ClusterRow),
      base_loss_prob : Float64,
      high_loss_prob : Float64,
    ) : Hash(String, Float64)
      result = {} of String => Float64
      cluster_rows.each do |mutation_id, row|
        prev = row.cellular_prevalence
        if prev
          result[mutation_id] = prev < base_loss_prob ? high_loss_prob : base_loss_prob
        end
      end
      result
    end

    private def self.compute_loss_log_prob(rows : Array(InputRow), config : PhyCloneRunConfig) : Float64
      return 0.0 unless config.assign_loss_prob? || config.user_provided_loss_prob?

      raw_prob = if config.user_provided_loss_prob?
                   probs = rows.compact_map(&.loss_prob)
                   if probs.size != rows.size
                     raise CliError.new("--user-provided-loss-prob requires loss_prob for all rows")
                   end
                   probs.sum / probs.size
                 else
                   representative = rows.find(&.chrom)
                   chrom = representative.try(&.chrom) || ""
                   high_loss = chrom == "X" || chrom == "Y" || chrom == "chrX" || chrom == "chrY"
                   high_loss ? config.high_loss_prob : config.loss_prob
                 end

      prob = raw_prob.clamp(1e-9, 1.0 - 1e-9)
      Math.log(prob)
    end

    private def self.build_sample_profiles(rows : Array(InputRow)) : Array(SampleProfile)
      totals = {} of String => {Int32, Int32}

      rows.each do |row|
        current = totals[row.sample_id]? || {0, 0}
        totals[row.sample_id] = {current[0] + row.ref_counts, current[1] + row.alt_counts}
      end

      sample_ids = totals.keys
      sample_ids.sort!
      sample_ids.map do |sample_id|
        ref_counts, alt_counts = totals[sample_id]
        SampleProfile.new(sample_id, ref_counts, alt_counts)
      end
    end

    private def self.write_trace(path : String, compress : Bool, contents : String)
      if compress
        File.open(path, "w") do |file|
          Compress::Gzip::Writer.open(file) do |gzip|
            gzip << contents
          end
        end
      else
        File.write(path, contents)
      end
    end

    private def self.best_trace_record(path : String) : TraceRecord?
      scan_trace(path).best_record
    end

    private def self.each_trace_record(path : String, & : TraceRecord ->)
      if path.ends_with?(".gz")
        File.open(path) do |file|
          Compress::Gzip::Reader.open(file) do |gzip|
            each_trace_record_io(gzip) do |record|
              yield record
            end
          end
        end
      else
        File.open(path) do |file|
          each_trace_record_io(file) do |record|
            yield record
          end
        end
      end
    end

    private def self.each_trace_record_io(io : IO, & : TraceRecord ->)
      io.each_line do |line|
        next if line.strip.empty?
        yield parse_trace_record(JSON.parse(line))
      end
    end

    private def self.parse_trace_record(document : JSON::Any) : TraceRecord
      validate_trace_document(document)

      log_p = document["log_p"].as_f
      log_p_one = document["log_p_one"]?.try(&.as_f?) || log_p

      TraceRecord.new(
        document["chain"].as_i,
        document["iter"].as_i,
        log_p,
        log_p_one,
        document["topology_id"].as_s,
        parse_trace_nodes(document),
        parse_trace_edges(document),
        parse_trace_clusters(document),
        parse_outlier_assignments(document),
        parse_trace_metadata(document),
      )
    end

    private def self.parse_trace_nodes(document : JSON::Any) : Array(TraceNode)
      nodes = document["tree"]["nodes"].as_a.map do |node|
        # Support both old format (cluster_id: Int) and new format (cluster_ids: Array(Int))
        cluster_ids = if arr = node["cluster_ids"]?.try(&.as_a?)
                        arr.compact_map(&.as_i?)
                      elsif single = node["cluster_id"]?.try(&.as_i?)
                        [single]
                      else
                        [] of Int32
                      end
        TraceNode.new(
          node["id"].as_s,
          node["kind"].as_s,
          cluster_ids
        )
      end
      nodes
    end

    private def self.parse_trace_edges(document : JSON::Any) : Array(TraceEdge)
      document["tree"]["edges"].as_a.map do |edge|
        TraceEdge.new(edge["parent"].as_s, edge["child"].as_s)
      end
    end

    private def self.parse_trace_clusters(document : JSON::Any) : Array(ClusterSummary)
      document["clusters"].as_a.map do |cluster|
        sample_profiles = cluster["sample_profiles"]?.try(&.as_a).try do |profiles|
          profiles.map do |profile|
            SampleProfile.new(
              profile["sample_id"].as_s,
              profile["ref_counts"].as_i,
              profile["alt_counts"].as_i,
            )
          end
        end || [] of SampleProfile

        outlier_data_points = cluster["outlier_data_points"]?.try(&.as_a).try do |points|
          points.map do |point|
            sample_observations = point["sample_observations"]?.try(&.as_a).try do |observations|
              observations.map do |observation|
                OutlierSampleObservation.new(
                  observation["sample_id"].as_s,
                  observation["ref_counts"].as_i,
                  observation["alt_counts"].as_i,
                  observation["major_cn"].as_i,
                  observation["minor_cn"].as_i,
                  observation["normal_cn"].as_i,
                  observation["tumour_content"].as_f,
                  observation["error_rate"].as_f,
                )
              end
            end

            OutlierDataPoint.new(
              point["name"].as_s,
              point["outlier_prob"].as_f,
              point["outlier_prob_not"].as_f,
              point["outlier_marginal_prob"].as_f,
              point["loss_log_prob"]?.try(&.as_f?) || 0.0,
              sample_observations || [] of OutlierSampleObservation,
            )
          end
        end

        if outlier_data_points.nil?
          # Backward compatibility for older traces that only had
          # cluster-level outlier aggregate terms.
          legacy_prior = cluster["outlier_log_prior"]?.try(&.as_f?) || 0.0
          legacy_marginal = cluster["outlier_log_marginal_prob"]?.try(&.as_f?) || 0.0
          outlier_data_points = [OutlierDataPoint.new(
            "cluster_#{cluster["cluster_id"].as_i}",
            legacy_prior,
            legacy_prior,
            legacy_marginal,
            0.0,
            [] of OutlierSampleObservation,
          )]
        end

        ClusterSummary.new(
          cluster["cluster_id"].as_i,
          cluster["mutation_ids"].as_a.map(&.as_s),
          cluster["sample_ids"].as_a.map(&.as_s),
          sample_profiles,
          outlier_data_points,
        )
      end
    end

    private def self.parse_outlier_assignments(document : JSON::Any) : Array(OutlierAssignment)
      document["outlier_assignments"]?.try(&.as_a).try do |items|
        items.map do |item|
          # log_odds_outlier_vs_in_tree is optional (omitted when not computed by unclone)
          log_odds = item["log_odds_outlier_vs_in_tree"]?.try(&.as_f?)
          OutlierAssignment.new(
            item["mutation_id"].as_s,
            item["is_outlier"].as_bool,
            log_odds,
          )
        end
      end || [] of OutlierAssignment
    end

    private def self.parse_trace_metadata(document : JSON::Any) : TraceMetadata?
      document["metadata"]?.try do |meta|
        TraceMetadata.new(
          meta["num_chains"].as_i,
          meta["num_iters"].as_i,
          meta["seed"]?.try(&.as_i64?).try(&.to_u64),
        )
      end
    end

    private def self.validate_trace_document(document : JSON::Any)
      schema_version = document["schema_version"].as_i
      if schema_version != TRACE_SCHEMA_VERSION
        raise CliError.new("Unsupported phy trace schema_version: #{schema_version} (expected #{TRACE_SCHEMA_VERSION})")
      end

      nodes = document["tree"]["nodes"].as_a
      if nodes.empty?
        raise CliError.new("Invalid phy trace record: tree.nodes must not be empty")
      end

      root_nodes = nodes.select do |node|
        node["id"].as_s == TRACE_ROOT_ID && node["kind"].as_s == TRACE_ROOT_KIND
      end
      if root_nodes.size != 1
        raise CliError.new("Invalid phy trace record: expected exactly one root node")
      end

      cluster_node_ids = Set(String).new
      nodes.each do |node|
        next unless node["kind"].as_s == "cluster"
        cluster_node_ids << node["id"].as_s
      end

      clusters = document["clusters"].as_a
      # Validate that tree nodes do not exceed cluster array size.
      # (Tree may contain fewer nodes if SMC did not generate all clusters.)
      if cluster_node_ids.size > clusters.size
        raise CliError.new("Invalid phy trace record: cluster node count (#{cluster_node_ids.size}) exceeds clusters array size (#{clusters.size})")
      end

      document["tree"]["edges"].as_a.each do |edge|
        child_id = edge["child"].as_s
        next if child_id == TRACE_ROOT_ID
        unless cluster_node_ids.includes?(child_id)
          raise CliError.new("Invalid phy trace record: edge child '#{child_id}' is not a known cluster node")
        end
      end
    end

    private def self.to_newick(record : TraceRecord) : String
      children_by_parent = Hash(String, Array(String)).new { |hash, key| hash[key] = [] of String }
      record.edges.each do |edge|
        children_by_parent[edge.parent] << edge.child
      end

      build_newick_subtree("root", children_by_parent, record) + ";"
    end

    private def self.build_newick_subtree(node_id : String, children_by_parent : Hash(String, Array(String)), record : TraceRecord) : String
      children = children_by_parent[node_id]? || [] of String
      label = node_label(node_id, record)
      return label if children.empty?
      "(" + children.sort.map { |child| build_newick_subtree(child, children_by_parent, record) }.join(",") + ")#{label}"
    end

    private def self.node_label(node_id : String, record : TraceRecord) : String
      node = record.nodes.find { |value| value.id == node_id }
      return node_id unless node
      return "root" if node.kind == "root"
      # Use first cluster_id for label; multi-cluster nodes show all ids joined by "_"
      ids = node.cluster_ids
      ids.empty? ? node_id : "cluster_#{ids.join("_")}"
    end
  end
end
