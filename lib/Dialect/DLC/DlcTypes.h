#ifndef LIB_DIALECT_DLC_DLCTYPES
#define LIB_DIALECT_DLC_DLCTYPES

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/ADTExtras.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/DLC/DlcTypes.h.inc"

namespace mlir {
namespace dlc {

} // end dlc
} // end mlir
#endif /* LIB_DIALECT_DLC_DLCTYPES */