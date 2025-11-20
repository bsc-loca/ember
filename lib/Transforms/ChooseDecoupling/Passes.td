//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_CHOOSEDECOUPLING_TD_
#define LIB_TRANSFORMS_CHOOSEDECOUPLING_TD_

include "mlir/Pass/PassBase.td"


def ChooseDecoupling : Pass<"choose-decoupling", "mlir::func::FuncOp"> {
  let summary = "Choose operation decoupling (among scf, arith, and memref)";
  let description = [{
    A pass that inserts (1) SLC primitives to mark loop offloading and (2) callbacks to split operations running on SLC and core.
  }];

  let dependentDialects = ["mlir::slc::SlcDialect", "mlir::slcvec::SlcVecDialect"];
}


#endif // LIB_TRANSFORMS_CHOOSEDECOUPLING_TD_