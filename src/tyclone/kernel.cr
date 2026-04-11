module Tyclone
  module Kernel
    def self.handle_result(result_ptr : KernelAbi::Result*, error_ptr : KernelAbi::Error*, rc : Int32) : PcvTabularResult
      if rc != 0
        message = "Unknown kernel error"
        unless error_ptr.null?
          message_ptr = LibPcv.pcv_error_message(error_ptr)
          message = String.new(message_ptr) unless message_ptr.null?
          LibPcv.pcv_error_free(error_ptr)
        end
        raise KernelError.new(message)
      end

      PcvTabularResult.new(result_ptr)
    end

    def self.fit(config : ViConfig, rows : Array(IndexedRow), num_mutations : Int32, num_samples : Int32) : PcvTabularResult
      ViKernel.fit(config, KernelAbi.build_rows(rows), num_mutations, num_samples)
    end
  end
end
