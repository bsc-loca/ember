#ifndef LIB_TRANSFORMS_BUFFERCOMPOUNDTYPES_PASSES_H_
#define LIB_TRANSFORMS_BUFFERCOMPOUNDTYPES_PASSES_H_

#include "mlir/include/mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace slc {

std::unique_ptr<Pass> createBufferCompoundTypes();

#define GEN_PASS_DECL
#include "lib/Transforms/BufferCompoundTypes/Passes.h.inc"

std::unique_ptr<mlir::Pass> createBufferCompoundTypes();

} // namespace slc
} // namespace mlir

#endif // LIB_TRANSFORMS_BUFFERCOMPOUNDTYPES_PASSES_H_