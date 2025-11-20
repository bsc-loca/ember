module {
  func.func @sls_scf(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg2: !llvm.struct<(array<2 x i64>, array<3 x i64>)>, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>) -> memref<?x?xf64> {
    %c0 = arith.constant 0 : index
    %0 = slc.to_stream %c0 : index to !slc.stream<index>
    %c1 = arith.constant 1 : index
    %1 = slc.to_stream %c1 : index to !slc.stream<index>
    %2 = llvm.extractvalue %arg2[0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)> 
    %3 = arith.index_cast %2 : i64 to index
    %4 = slc.to_stream %3 : index to !slc.stream<index>
    %dim = memref.dim %arg3, %c1 : memref<?x?xf64>
    %5 = slc.to_stream %dim : index to !slc.stream<index>
    slc.for %arg5 = %0 to %4 step %1  : !slc.stream<index> -> !slc.stream<index> {
      %6 = slc.mem_stream %arg0[%arg5] : memref<?xindex>[!slc.stream<index>] into !slc.stream<index>
      %7 = slc.alu_stream< add> %arg5, %1 : !slc.stream<index>, !slc.stream<index> -> !slc.stream<index>
      %8 = slc.mem_stream %arg0[%7] : memref<?xindex>[!slc.stream<index>] into !slc.stream<index>
      slc.for %arg6 = %6 to %8 step %1  : !slc.stream<index> -> !slc.stream<index> {
        %9 = slc.mem_stream %arg1[%arg6] : memref<?xindex>[!slc.stream<index>] into !slc.stream<index>
        slc.for %arg7 = %0 to %5 step %1  : !slc.stream<index> -> !slc.stream<index> {
          slcvec.vectorize_for
          %10 = slc.mem_stream %arg3[%9, %arg7] : memref<?x?xf64>[!slc.stream<index>, !slc.stream<index>] into !slc.stream<f64>
          slc.callback {
            %11 = slc.from_stream %arg5 : !slc.stream<index> to index
            %12 = slc.from_stream %arg7 : !slc.stream<index> to index
            %13 = slc.from_stream %10 : !slc.stream<f64> to f64
            %14 = memref.load %arg4[%11, %12] : memref<?x?xf64>
            %15 = arith.addf %14, %13 : f64
            memref.store %15, %arg4[%11, %12] : memref<?x?xf64>
          } : index 
          slc.yield
        }
        slc.yield
      }
      slc.yield
    }
    return %arg4 : memref<?x?xf64>
  }
}