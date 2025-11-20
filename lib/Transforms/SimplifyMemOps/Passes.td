//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_SIMPLIFYMEMOPS_TD_
#define LIB_TRANSFORMS_SIMPLIFYMEMOPS_TD_

include "mlir/Pass/PassBase.td"


def SimplifyMemOps : Pass<"simplify-mem-ops", "mlir::func::FuncOp"> {
  let summary = "Simplify gather scatter operations into vector loads";
  let description = [{
    Simplify gather scatter operations into vector loads.
  }];

  let dependentDialects = ["mlir::slc::SlcDialect", "mlir::slc::SlcVecDialect", "mlir::vector::VectorDialect"];
}


#endif // LIB_TRANSFORMS_SIMPLIFYMEMOPS_TD_