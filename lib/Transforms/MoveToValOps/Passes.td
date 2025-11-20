//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_MOVETOVALOPS_TD_
#define LIB_TRANSFORMS_MOVETOVALOPS_TD_

include "mlir/Pass/PassBase.td"


def MoveToValOps : Pass<"move-to-val-ops", "mlir::func::FuncOp"> {
  let summary = "Move materialization to_val operations into callbacks";
  let description = [{
    A pass that moves stream to value materialization operations into SLC callbacks.
  }];

  let dependentDialects = ["mlir::slc::SlcDialect"];
}


#endif // LIB_TRANSFORMS_MOVETOVALOPS_TD_