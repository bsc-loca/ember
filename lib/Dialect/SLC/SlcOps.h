#ifndef LIB_DIALECT_SLC_SLCOPS
#define LIB_DIALECT_SLC_SLCOPS

#include "lib/Dialect/SLC/SlcDialect.h"
#include "lib/Dialect/SLC/SlcTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h" // from @llvm-project
#include "mlir/include/mlir/IR/Dialect.h"      // from @llvm-project

#include "mlir/include/mlir/IR/TypeUtilities.h"

#define GET_OP_CLASSES
#include "lib/Dialect/SLC/SlcOps.h.inc"

#endif /* LIB_DIALECT_SLC_SLCOPS */
