#include "lib/Dialect/SLCVEC/SlcVecDialect.h"
#include "lib/Dialect/SLCVEC/SlcVecOps.h"
#include "lib/Dialect/SLCVEC/SlcVecTypes.h"

#include "mlir/include/mlir/IR/Builders.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/SLCVEC/SlcVecDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/SLCVEC/SlcVecTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/SLCVEC/SlcVecOps.cpp.inc"

#include "lib/Dialect/SLCVEC/SlcVecEnumDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/SLCVEC/SlcVecAttrDefs.cpp.inc"

namespace mlir::slcvec {

void SlcVecDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/SLCVEC/SlcVecTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/SLCVEC/SlcVecOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/SLCVEC/SlcVecAttrDefs.cpp.inc"
      >();
}

} // namespace mlir::slcvec
