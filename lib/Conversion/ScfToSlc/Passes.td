//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CONVERSION_SCFTOSLC_TD_
#define LIB_CONVERSION_SCFTOSLC_TD_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ScfToSlc
//===----------------------------------------------------------------------===//

def ScfToSlc : Pass<"scf-to-slc"/*, "func::FuncOp"*/> {
  let summary = "Convert operations from SCF dialect to the SLC dialect";
  let constructor = "mlir::createScfToSlcPass()";
  let dependentDialects = [
    "arith::ArithDialect", 
    "func::FuncDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "slc::SlcDialect",
  ];
}

#endif // LIB_CONVERSION_SCFTOSLC_TD_