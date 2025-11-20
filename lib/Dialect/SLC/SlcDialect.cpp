#include "lib/Dialect/SLC/SlcDialect.h"
#include "lib/Dialect/SLC/SlcOps.h"
#include "lib/Dialect/SLC/SlcTypes.h"

#include "mlir/include/mlir/IR/Builders.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"

#include "lib/Dialect/SLC/SlcDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/SLC/SlcTypes.cpp.inc"

#define GET_OP_CLASSES
#include "lib/Dialect/SLC/SlcOps.cpp.inc"

#include "lib/Dialect/SLC/SlcEnumDefs.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "lib/Dialect/SLC/SlcAttrDefs.cpp.inc"

namespace mlir::slc {

void SlcDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "lib/Dialect/SLC/SlcTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "lib/Dialect/SLC/SlcOps.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "lib/Dialect/SLC/SlcAttrDefs.cpp.inc"
      >();
}

// SlcStreamEncodingAttr getSlcStreamEncoding(Type type) {
//   if (auto ttp = llvm::dyn_cast<RankedTensorType>(type))
//     return llvm::dyn_cast_or_null<SlcStreamEncodingAttr>(ttp.getEncoding());
//   // if (auto mdtp = llvm::dyn_cast<StorageSpecifierType>(type))
//   //   return mdtp.getEncoding();
//   return nullptr;
// }

} // namespace mlir::slc
