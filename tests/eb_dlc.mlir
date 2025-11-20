module {
  func.func @sls_scf(%arg0: memref<?xindex>, %arg1: memref<?xindex>, %arg2: !llvm.struct<(array<2 x i64>, array<3 x i64>)>, %arg3: memref<?x?xf64>, %arg4: memref<?x?xf64>) -> memref<?x?xf64> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<0.000000e+00> : vector<8xf64>
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %cst_0 = arith.constant dense<true> : vector<8xi1>
    %0 = slc.to_stream %c0 : index to !slc.stream<index>
    %1 = slc.to_stream %c1 : index to !slc.stream<index>
    %2 = slc.to_stream %c8 : index to !slc.stream<index>
    %3 = llvm.extractvalue %arg2[0, 0] : !llvm.struct<(array<2 x i64>, array<3 x i64>)> 
    %4 = arith.index_cast %3 : i64 to index
    %5 = slc.to_stream %4 : index to !slc.stream<index>
    %dim = memref.dim %arg3, %c1 : memref<?x?xf64>
    %6 = slc.to_stream %dim : index to !slc.stream<index>
    %7 = slc.to_stream %cst_0 : vector<8xi1> to !slc.stream<vector<8xi1>>
    dlc.configure_access_engine {
      %10 = dlc.new_tu %0 to %5 step %1 : !slc.stream<index> into !dlc.tu
      %11 = dlc.get_ind_var %10 : !slc.stream<index>
      %12 = slc.mem_stream %arg0[%11] : memref<?xindex>[!slc.stream<index>] into !slc.stream<index>
      %13 = slc.alu_stream< add> %11, %1 : !slc.stream<index>, !slc.stream<index> -> !slc.stream<index>
      %14 = slc.mem_stream %arg0[%13] : memref<?xindex>[!slc.stream<index>] into !slc.stream<index>
      %15 = dlc.new_tu %12 to %14 step %1 : !slc.stream<index> into !dlc.tu
      %16 = dlc.get_ind_var %15 : !slc.stream<index>
      %17 = slc.mem_stream %arg1[%16] : memref<?xindex>[!slc.stream<index>] into !slc.stream<index>
      %18 = dlc.new_tu %0 to %6 step %2 : !slc.stream<index> into !dlc.tu
      %19 = dlc.get_ind_var %18 : !slc.stream<index>
      %20 = slcvec.mem_stream %arg3[%17, %19] : memref<?x?xf64>[!slc.stream<index>, !slc.stream<index>] into !slc.stream<vector<8xf64>>
      dlc.register_operand %18,  end, %20 : !dlc.tu, !slc.stream<vector<8xf64>>
      dlc.register_callback %15,  end {id = 4 : i64} : !dlc.tu
      dlc.register_callback %10,  end {id = 1 : i64} : !dlc.tu
    } : 
    %8 = dlc.compute_loop -> index 
    case 1 {
      dlc.increment_var
      dlc.yield
    }
    case 4 {
      %10 = dlc.pop_operand : memref<?xf64>
      scf.for %arg5 = %c0 to %dim step %c1 {
        %11 = slcvec.mask %arg5 %c0 %dim %c1 : vector<8xi1>
        %12 = slcvec.from_buffer %10[%arg5] : memref<?xf64>[index] -> vector<8xf64>
        %13 = dlc.get_var : index
        %14 = vector.maskedload %arg4[%13, %arg5], %11, %cst : memref<?x?xf64>, vector<8xi1>, vector<8xf64> into vector<8xf64>
        %15 = arith.addf %14, %12 : vector<8xf64>
        %16 = dlc.get_var : index
        vector.maskedstore %arg4[%16, %arg5], %11, %15 : memref<?x?xf64>, vector<8xi1>, vector<8xf64>
      }
      dlc.yield
    }
    return %arg4 : memref<?x?xf64>
  }
}