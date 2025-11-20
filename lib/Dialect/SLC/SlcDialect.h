#ifndef LIB_DIALECT_SLC_SLCDIALECT
#define LIB_DIALECT_SLC_SLCDIALECT

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"

#include "lib/Dialect/SLC/SlcDialect.h.inc"

#include "lib/Dialect/SLC/SlcEnumDefs.h.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/SLC/SlcAttrDefs.h.inc"

namespace mlir::slc {
// SlcStreamEncodingAttr getSlcStreamEncoding(Type type);
}

#endif /* LIB_DIALECT_SLC_SLCDIALECT */
