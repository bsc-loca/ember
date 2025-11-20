#include "lib/Dialect/SLC/SlcTypes.h"

// #include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::slc;
/*
SlcVecStreamType
SlcVecStreamType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                            Type elementType) const {
  SlcVecStreamType::Builder builder(llvm::cast<SlcVecStreamType>(*this));
  if (shape)
    builder.setShape(*shape);
  builder.setElementType(elementType);
  return builder;
}

LogicalResult
SlcVecStreamType::verify(function_ref<InFlightDiagnostic()> emitError,
                         ArrayRef<int64_t> shape, Type elementType) {
  if (!isValidElementType(elementType))
    return emitError() << "invalid memref element type";

  // Negative sizes are not allowed except for `kDynamic`.
  for (int64_t s : shape)
    if (s < 0 && !ShapedType::isDynamic(s))
      return emitError() << "invalid memref size";

  return success();
}

Type SlcVecStreamType::parse(AsmParser &odsParser) {
  assert(false); // TODO check AsmParser ParseMemRefType();
}

void SlcVecStreamType::print(AsmPrinter &odsPrinter) const {
  odsPrinter.getStream() << "<";
  odsPrinter.printDimensionList(getShape());
  odsPrinter.getStream() << "x";
  odsPrinter.printType(getElementType());
  odsPrinter.getStream() << ">";
}
*/
/*
SlcBufferType SlcBufferType::cloneWith(std::optional<ArrayRef<int64_t>> shape,
                                       Type elementType) const {
  SlcBufferType::Builder builder(llvm::cast<SlcBufferType>(*this));
  if (shape)
    builder.setShape(*shape);
  builder.setElementType(elementType);
  return builder;
}

LogicalResult
SlcBufferType::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<int64_t> shape, Type elementType) {
  if (!isValidElementType(elementType))
    return emitError() << "invalid memref element type";

  // Negative sizes are not allowed except for `kDynamic`.
  for (int64_t s : shape)
    if (s < 0 && !ShapedType::isDynamic(s))
      return emitError() << "invalid memref size";

  return success();
}

Type SlcBufferType::parse(AsmParser &odsParser) {
  assert(false); // TODO check AsmParser ParseMemRefType();
}

void SlcBufferType::print(AsmPrinter &odsPrinter) const {
  odsPrinter.printDimensionList(getShape());
  assert(false);
}
*/