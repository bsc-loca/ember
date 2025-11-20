#include "lib/Dialect/SLCVEC/SlcVecTypes.h"

// #include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::slcvec;

SlcStreamVecType
SlcStreamVecType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                            Type elementType) const {
  SlcStreamVecType::Builder builder(llvm::cast<SlcStreamVecType>(*this));
  if (shape)
    builder.setShape(*shape);
  builder.setElementType(elementType);
  return builder;
}

LogicalResult
SlcStreamVecType::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<int64_t> shape, Type elementType) {
  if (!isValidElementType(elementType))
    return emitError() << "invalid memref element type";

  // Negative sizes are not allowed except for `kDynamic`.
  for (int64_t s : shape)
    if (s < 0 && !ShapedType::isDynamic(s))
      return emitError() << "invalid memref size";

  return success();
}

Type SlcStreamVecType::parse(AsmParser &odsParser) {
  assert(false); // TODO check AsmParser ParseMemRefType();
}

void SlcStreamVecType::print(AsmPrinter &odsPrinter) const {
  odsPrinter.printDimensionList(getShape());
  assert(false);
}