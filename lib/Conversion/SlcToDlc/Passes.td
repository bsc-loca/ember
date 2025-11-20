//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CONVERSION_SLCTODLC_TD_
#define LIB_CONVERSION_SLCTODLC_TD_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SlcToDlc
//===----------------------------------------------------------------------===//

def SlcToDlc : Pass<"slc-to-dlc", "func::FuncOp"> {
  let summary = "Convert operations from SLC dialect to the DLC dialect";
  let constructor = "mlir::createSlcToDlcPass()";
  let dependentDialects = [
    "arith::ArithDialect", 
    "func::FuncDialect",
    "memref::MemRefDialect",
    "slc::SlcDialect",
    "slcvec::SlcVecDialect",
    "dlc::DlcDialect",
  ];
}

#endif // LIB_CONVERSION_SLCTODLC_TD_