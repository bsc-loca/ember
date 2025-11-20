#ifndef LIB_DIALECT_DLC_DLCOPS
#define LIB_DIALECT_DLC_DLCOPS

#include "lib/Dialect/DLC/DlcDialect.h"
#include "lib/Dialect/DLC/DlcTypes.h"
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


/// Parse the case regions and values.
using namespace mlir;
static ParseResult
parseSwitchCases(OpAsmParser &p, DenseI64ArrayAttr &cases,
                 SmallVectorImpl<std::unique_ptr<Region>> &caseRegions) {
  SmallVector<int64_t> caseValues;
  while (succeeded(p.parseOptionalKeyword("case"))) {
    int64_t value;
    Region &region = *caseRegions.emplace_back(std::make_unique<Region>());
    if (p.parseInteger(value) || p.parseRegion(region, /*arguments=*/{}))
      return failure();
    caseValues.push_back(value);
  }
  cases = p.getBuilder().getDenseI64ArrayAttr(caseValues);
  return success();
}

/// Print the case regions and values.
static void printSwitchCases(OpAsmPrinter &p, Operation *op,
                             DenseI64ArrayAttr cases, RegionRange caseRegions) {
  for (auto [value, region] : llvm::zip(cases.asArrayRef(), caseRegions)) {
    p.printNewline();
    p << "case " << value << ' ';
    p.printRegion(*region, /*printEntryBlockArgs=*/false);
  }
}

#define GET_OP_CLASSES
#include "lib/Dialect/DLC/DlcOps.h.inc"

#endif /* LIB_DIALECT_DLC_DLCOPS */
