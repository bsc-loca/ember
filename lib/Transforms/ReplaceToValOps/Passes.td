//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_REPLACETOVALOPS_TD_
#define LIB_TRANSFORMS_REPLACETOVALOPS_TD_

include "mlir/Pass/PassBase.td"


def ReplaceToValOps : Pass<"replace-to-val-ops", "mlir::func::FuncOp"> {
  let summary = "Replace simple materialization to_val operations with induction variables";
  let description = [{
    A pass that replaces stream to value materialization operations with induction variables.
  }];

  let dependentDialects = ["mlir::slc::SlcDialect"];
}


#endif // LIB_TRANSFORMS_REPLACETOVALOPS_TD_