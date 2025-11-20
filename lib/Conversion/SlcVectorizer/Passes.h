//===- Passes.h - Utils to convert from the linalg dialect ----------===//
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CONVERSION_SLCVECTORIZER_PASSES_H_
#define LIB_CONVERSION_SLCVECTORIZER_PASSES_H_

#include "lib/Conversion/SlcVectorizer/SlcVectorizer.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "lib/Conversion/SlcVectorizer/Passes.h.inc"

} // namespace mlir

#endif // LIB_CONVERSION_SLCVECTORIZER_PASSES_H