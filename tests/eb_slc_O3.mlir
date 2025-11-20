module {
  func.func @sls_scf(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg2: !llvm.struct<(array<2 x i64>, array<3 x i64>)>, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>) -> memref<?x?xf64> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<8xf64>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %mask = arith.constant dense<true> : vector<8xi1>
    %c0s = slc.to_stream %c0 : index to !slc.stream<index>
    %c1s = slc.to_stream %c1 : index to !slc.stream<index>
    %c8s = slc.to_stream %c8 : index to !slc.stream<index>
    %1 = llvm.extractvalue %arg2[0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)> 
    %2 = arith.index_cast %1 : i64 to index
    %c2s = slc.to_stream %2 : index to !slc.stream<index>
    %dim = memref.dim %arg3, %c1 : memref<?x?xf64>
    %dims = slc.to_stream %dim : index to !slc.stream<index>
    %masks = slc.to_stream %mask : vector<8xi1> to !slc.stream<vector<8xi1>>
    %3 = slc.for %arg5 = %c0s to %c2s step %c1s init_args(%arg6 = %c0) -> (index)  : !slc.stream<index> -> !slc.stream<index> {
      %4 = slc.mem_stream %arg0[%arg5] : memref<?xindex>[!slc.stream<index>] into !slc.stream<index>
      %5 = slc.alu_stream< add> %arg5, %c1s : !slc.stream<index>, !slc.stream<index> -> !slc.stream<index>
      %6 = slc.mem_stream %arg0[%5] : memref<?xindex>[!slc.stream<index>] into !slc.stream<index>
      slc.for %arg7 = %4 to %6 step %c1s  : !slc.stream<index> -> !slc.stream<index> {
        %8 = slc.mem_stream %arg1[%arg7] : memref<?xindex>[!slc.stream<index>] into !slc.stream<index>
        %9 = slcvec.buf_stream : !slc.stream<memref<?xf64>>
        slcvec.for<8>(%arg8 xx %arg9) = %c0s to %dims step %c8s mask %masks  : (!slc.stream<index> xx !slc.stream<vector<8xi1>>) -> (!slc.stream<vector<8xindex>> xx !slc.stream<vector<8xi1>>) {
          %10 = slcvec.mem_stream %arg3[%8, %arg8] : memref<?x?xf64>[!slc.stream<index>, !slc.stream<vector<8xindex>>] into !slc.stream<vector<8xf64>>
          slcvec.to_buffer %10, %9[%arg8] : !slc.stream<vector<8xf64>> -> !slc.stream<memref<?xf64>>[!slc.stream<vector<8xindex>>]
          slcvec.yield
        } {loopConfig = #slcvec<loop_config bcast>}//, vectorLength = 8 : index}
        slc.callback {
          %10 = slc.from_stream %9 : !slc.stream<memref<?xf64>> to memref<?xf64>
          scf.for %arg8 = %c0 to %dim step %c1 {
            %11 = slcvec.mask %arg8 %c0 %dim %c1 : vector<8xi1>
            %12 = slcvec.from_buffer %10[%arg8] : memref<?xf64>[index] -> vector<8xf64>
            %13 = vector.maskedload %arg4[%arg6, %arg8], %11, %cst : memref<?x?xf64>, vector<8xi1>, vector<8xf64> into vector<8xf64>
            %14 = arith.addf %13, %12 : vector<8xf64>
            vector.maskedstore %arg4[%arg6, %arg8], %11, %14 : memref<?x?xf64>, vector<8xi1>, vector<8xf64>
          }
        } : index
        slc.yield
      }
      %7 = slc.callback {
        %8 = arith.addi %arg6, %c1 : index
        slc.yield %8 : index
      } : index
      slc.yield %7 : index
    }
    return %arg4 : memref<?x?xf64>
  }
}