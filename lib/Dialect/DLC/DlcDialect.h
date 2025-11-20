#ifndef LIB_DIALECT_DLC_DLCDIALECT
#define LIB_DIALECT_DLC_DLCDIALECT

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"

#include "lib/Dialect/DLC/DlcDialect.h.inc"

#include "lib/Dialect/DLC/DlcEnumDefs.h.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/DLC/DlcAttrDefs.h.inc"

namespace mlir::dlc {
// DlcStreamEncodingAttr getDlcStreamEncoding(Type type);
}

#endif /* LIB_DIALECT_DLC_DLCDIALECT */
