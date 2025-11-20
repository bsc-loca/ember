#ifndef LIB_DIALECT_SLCVEC_SLCVECOPS
#define LIB_DIALECT_SLCVEC_SLCVECOPS

#include "lib/Dialect/SLCVEC/SlcVecDialect.h"
#include "lib/Dialect/SLCVEC/SlcVecTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"      // from @llvm-project

#include "mlir/include/mlir/IR/TypeUtilities.h"

#include "lib/Dialect/SLC/SlcOps.h"

#define GET_OP_CLASSES
#include "lib/Dialect/SLCVEC/SlcVecOps.h.inc"

#endif /* LIB_DIALECT_SLCVEC_SLCVECOPS */
