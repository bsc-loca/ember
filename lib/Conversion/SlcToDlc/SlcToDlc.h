//===- SlcToDlc.h - Utils to convert from the linalg dialect ----------===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CONVERSION_SLCTODLC_H_
#define LIB_CONVERSION_SLCTODLC_H_

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {

/// Create a pass to convert SLC operations to the DLC dialect.
std::unique_ptr<mlir::Pass> createSlcToDlcPass();

} // namespace mlir

#endif // LIB_CONVERSION_SLCTODLC_H