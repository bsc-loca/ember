#include "lib/Dialect/DLC/DlcDialect.h"
#include "lib/Dialect/DLC/DlcOps.h"
#include "lib/Dialect/DLC/DlcTypes.h"

#include "mlir/include/mlir/IR/Builders.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/DLC/DlcDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/DLC/DlcTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/DLC/DlcOps.cpp.inc"

#include "lib/Dialect/DLC/DlcEnumDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/DLC/DlcAttrDefs.cpp.inc"

namespace mlir::dlc {

void DlcDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/DLC/DlcTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/DLC/DlcOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/DLC/DlcAttrDefs.cpp.inc"
      >();
}

} // namespace mlir::dlc
