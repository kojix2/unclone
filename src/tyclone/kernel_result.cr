module Tyclone
  abstract class KernelResultHandle
    abstract def free
  end

  abstract class TabularKernelResult < KernelResultHandle
    abstract def num_mutations : Int32
    abstract def num_samples : Int32
    abstract def num_clusters : Int32
    abstract def num_saved_trace_samples : Int32
    abstract def mutation_cluster_ids : Slice(Int32)
    abstract def mutation_cluster_probs : Slice(Float64)
    abstract def mutation_sample_prevalence : Slice(Float64)
    abstract def mutation_sample_prevalence_std : Slice(Float64)
    abstract def saved_mutation_sample_prevalence : Slice(Float64)
    abstract def saved_precision_trace : Slice(Float64)
    abstract def cluster_sample_prevalence : Slice(Float64)
    abstract def cluster_sample_prevalence_std : Slice(Float64)
  end

  class PcvTabularResult < TabularKernelResult
    def initialize(@ptr : KernelAbi::Result*)
    end

    def num_mutations : Int32
      LibPcv.pcv_result_num_mutations(@ptr).to_i32
    end

    def num_samples : Int32
      LibPcv.pcv_result_num_samples(@ptr).to_i32
    end

    def num_clusters : Int32
      LibPcv.pcv_result_num_clusters(@ptr).to_i32
    end

    def num_saved_trace_samples : Int32
      LibPcv.pcv_result_num_saved_trace_samples(@ptr).to_i32
    end

    def mutation_cluster_ids : Slice(Int32)
      ptr = LibPcv.pcv_result_mutation_cluster_ids(@ptr)
      Slice.new(ptr, num_mutations)
    end

    def mutation_cluster_probs : Slice(Float64)
      ptr = LibPcv.pcv_result_mutation_cluster_probs(@ptr)
      Slice.new(ptr, num_mutations)
    end

    def mutation_sample_prevalence : Slice(Float64)
      ptr = LibPcv.pcv_result_mutation_sample_prevalence(@ptr)
      Slice.new(ptr, num_mutations * num_samples)
    end

    def mutation_sample_prevalence_std : Slice(Float64)
      ptr = LibPcv.pcv_result_mutation_sample_prevalence_std(@ptr)
      Slice.new(ptr, num_mutations * num_samples)
    end

    def saved_mutation_sample_prevalence : Slice(Float64)
      ptr = LibPcv.pcv_result_saved_mutation_sample_prevalence(@ptr)
      Slice.new(ptr, num_saved_trace_samples * num_mutations * num_samples)
    end

    def saved_precision_trace : Slice(Float64)
      ptr = LibPcv.pcv_result_saved_precision_trace(@ptr)
      Slice.new(ptr, num_saved_trace_samples)
    end

    def cluster_sample_prevalence : Slice(Float64)
      ptr = LibPcv.pcv_result_cluster_sample_prevalence(@ptr)
      Slice.new(ptr, num_clusters * num_samples)
    end

    def cluster_sample_prevalence_std : Slice(Float64)
      ptr = LibPcv.pcv_result_cluster_sample_prevalence_std(@ptr)
      Slice.new(ptr, num_clusters * num_samples)
    end

    def free
      LibPcv.pcv_result_free(@ptr)
    end
  end
end
