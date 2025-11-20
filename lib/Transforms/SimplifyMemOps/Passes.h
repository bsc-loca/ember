#ifndef LIB_TRANSFORMS_SIMPLIFYMEMOPS_PASSES_H_
#define LIB_TRANSFORMS_SIMPLIFYMEMOPS_PASSES_H_

#include "mlir/include/mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace slc {

std::unique_ptr<Pass> createSimplifyMemOps();

#define GEN_PASS_DECL
#include "lib/Transforms/SimplifyMemOps/Passes.h.inc"

std::unique_ptr<mlir::Pass> createSimplifyMemOpsPass();

} // namespace slc
} // namespace mlir

#endif // LIB_TRANSFORMS_SIMPLIFYMEMOPS_PASSES_H_