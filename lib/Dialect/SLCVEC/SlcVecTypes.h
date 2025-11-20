#ifndef LIB_DIALECT_SLCVEC_SLCVECTYPES
#define LIB_DIALECT_SLCVEC_SLCVECTYPES

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/ADTExtras.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/SLCVEC/SlcVecTypes.h.inc"

namespace mlir {
namespace slcvec {

//===----------------------------------------------------------------------===//
// SlcStreamVecType
//===----------------------------------------------------------------------===//

/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class SlcStreamVecType::Builder {
public:
  // Build from another SlcStreamVecType.
  explicit Builder(SlcStreamVecType other)
      : shape(other.getShape()), elementType(other.getElementType()) {}

  // Build from scratch.
  Builder(ArrayRef<int64_t> shape, Type elementType)
      : shape(shape), elementType(elementType) {}

  Builder &setShape(ArrayRef<int64_t> newShape) {
    shape = newShape;
    return *this;
  }

  Builder &setElementType(Type newElementType) {
    elementType = newElementType;
    return *this;
  }

  operator SlcStreamVecType() {
    return SlcStreamVecType::get(shape, elementType);
  }

private:
  ArrayRef<int64_t> shape;
  Type elementType;
};

} // namespace slcvec
} // namespace mlir
#endif /* LIB_DIALECT_SLCVEC_SLCVECTYPES */