//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_CALLBACKVECTORIZER_TD_
#define LIB_TRANSFORMS_CALLBACKVECTORIZER_TD_

include "mlir/Pass/PassBase.td"


def CallbackVectorizer : Pass<"callback-vectorizer", "mlir::func::FuncOp"> {
  let summary = "Vectorizes callback content";
  let description = [{
    A pass that vectorizes callback code.
  }];

  let dependentDialects = ["mlir::slc::SlcDialect", "mlir::vector::VectorDialect"];
}


#endif // LIB_TRANSFORMS_CALLBACKVECTORIZER_TD_