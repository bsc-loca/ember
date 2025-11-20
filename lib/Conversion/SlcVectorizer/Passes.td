//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CONVERSION_SLCVECTORIZER_TD_
#define LIB_CONVERSION_SLCVECTORIZER_TD_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// SlcVectorizer
//===----------------------------------------------------------------------===//

def SlcVectorizer : Pass<"slc-vectorizer"/*, "func::FuncOp"*/> {
  let summary = "Vectorizes SLC dialect";
  let constructor = "mlir::createSlcVectorizerPass()";
  let dependentDialects = [
    "arith::ArithDialect", 
    "func::FuncDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "slc::SlcDialect",
    "slcvec::SlcVecDialect",
  ];
}

#endif // LIB_CONVERSION_SLCVECTORIZER_TD_