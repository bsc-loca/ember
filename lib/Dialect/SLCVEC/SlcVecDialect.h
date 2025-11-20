#ifndef LIB_DIALECT_SLCVEC_SLCVECDIALECT
#define LIB_DIALECT_SLCVEC_SLCVECDIALECT

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"

#include "lib/Dialect/SLCVEC/SlcVecDialect.h.inc"

#include "lib/Dialect/SLCVEC/SlcVecEnumDefs.h.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/SLCVEC/SlcVecAttrDefs.h.inc"

namespace mlir::slcvec {
// SlcStreamEncodingAttr getSlcStreamEncoding(Type type);
}

#endif /* LIB_DIALECT_SLCVEC_SLCVECDIALECT */
