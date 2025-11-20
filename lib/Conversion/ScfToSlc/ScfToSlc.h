//===- ScfToSlc.h - Utils to convert from the linalg dialect ----------===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CONVERSION_SCFTOSLC_H_
#define LIB_CONVERSION_SCFTOSLC_H_

#include <memory>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {

/// Create a pass to convert SCF operations to the SLC dialect.
std::unique_ptr<mlir::Pass> createScfToSlcPass();

} // namespace mlir

#endif // LIB_CONVERSION_SCFTOSLC_H