module {
  func.func @sls_scf(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg3: !llvm.struct<(array<2 x i64>, array<3 x i64>)>, %arg4: memref<?x?xf64>, %arg5: memref<?x?xf64>) -> memref<?x?xf64> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = llvm.extractvalue %arg3[0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)> 
    %1 = arith.index_cast %0 : i64 to index
    %dim = memref.dim %arg4, %c1 : memref<?x?xf64>
    scf.for %arg6 = %c0 to %1 step %c1 {
      %2 = memref.load %arg0[%arg6] : memref<?xindex>
      %3 = arith.addi %arg6, %c1 : index
      %4 = memref.load %arg0[%3] : memref<?xindex>
      scf.for %arg7 = %2 to %4 step %c1 {
        %5 = memref.load %arg1[%arg7] : memref<?xindex>
        scf.for %arg8 = %c0 to %dim step %c1 {
          %7 = memref.load %arg4[%5, %arg8] : memref<?x?xf64>
          %8 = memref.load %arg5[%arg6, %arg8] : memref<?x?xf64>
          %10 = arith.addf %8, %7 : f64
          memref.store %10, %arg5[%arg6, %arg8] : memref<?x?xf64>
        } {"Emitted from" = "linalg.generic"}
      } {"Emitted from" = "linalg.generic"}
    } {"Emitted from" = "linalg.generic"}
    return %arg5 : memref<?x?xf64>
  }
}