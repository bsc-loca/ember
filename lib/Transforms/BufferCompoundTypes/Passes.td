//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_TRANSFORMS_BUFFERCOMPOUNDTYPES_TD_
#define LIB_TRANSFORMS_BUFFERCOMPOUNDTYPES_TD_

include "mlir/Pass/PassBase.td"


def BufferCompoundTypes : Pass<"buffer-compound-types", "mlir::func::FuncOp"> {
  let summary = "Buffer marshaling of compound types";
  let description = [{
    A pass that buffers marshaling operations of compound types.
  }];

  let dependentDialects = ["mlir::slc::SlcDialect"];
}


#endif // LIB_TRANSFORMS_BUFFERCOMPOUNDTYPES_TD_