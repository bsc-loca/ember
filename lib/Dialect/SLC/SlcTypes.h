#ifndef LIB_DIALECT_SLC_SLCTYPES
#define LIB_DIALECT_SLC_SLCTYPES

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/Support/ADTExtras.h"
#include "mlir/include/mlir/IR/DialectImplementation.h"

#define GET_TYPEDEF_CLASSES
#include "lib/Dialect/SLC/SlcTypes.h.inc"

namespace mlir {
namespace slc {
    
//===--------------------------===// 
// A slc stream type
// class SlcStreamType : public Type {

//   /// The element type of the stream.
//   Type *ContainedType;
//   public: 
//     SlcStreamType(const SlcStreamType &) = delete;
//     SlcStreamType &operator=(const SlcStreamType &) = delete;
//     Type *getElementType() const { return ContainedType; }

//     /// This static method is the primary way to construct a SlcStreamType
//     static SlcStreamType *get(Type *ElementType);

//     /// Return true if the specified type is valid as a element type.
//     static bool isValidElementType(Type *ElemTy);

// };

//===--------------------------===//
// A slc stream type
/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class SlcStreamType::Builder {
public:
  /// Build from another VectorType.
  explicit Builder(SlcStreamType other) : elementType(other.getElementType()) {}

  /// Build from scratch.
  Builder(Type elementType) : elementType(elementType) {}

  Builder &setElementType(Type newElementType) {
    elementType = newElementType;
    return *this;
  }

    operator SlcStreamType() { return SlcStreamType::get(elementType); }

  private:
    Type elementType;
};
/*
//===----------------------------------------------------------------------===//
// SlcVecStreamType
//===----------------------------------------------------------------------===//

/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class SlcVecStreamType::Builder {
public:
  // Build from another SlcVecStreamType.
  explicit Builder(SlcVecStreamType other)
      : shape(other.getShape()), elementType(other.getElementType()) {}

  // Build from a VectorType.
  explicit Builder(VectorType vectorType)
      : shape(vectorType.getShape()), elementType(vectorType.getElementType()) {
  }

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

  operator SlcVecStreamType() {
    return SlcVecStreamType::get(shape, elementType);
  }

private:
  ArrayRef<int64_t> shape;
  Type elementType;
};
*/
/*
//===----------------------------------------------------------------------===//
// SlcBufferType
//===----------------------------------------------------------------------===//

/// This is a builder type that keeps local references to arguments. Arguments
/// that are passed into the builder must outlive the builder.
class SlcBufferType::Builder {
public:
  // Build from another SlcBufferType.
  explicit Builder(SlcBufferType other)
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

  operator SlcBufferType() { return SlcBufferType::get(shape, elementType); }

private:
  ArrayRef<int64_t> shape;
  Type elementType;
};
*/

} // end slc
} // end mlir
#endif /* LIB_DIALECT_SLC_SLCTYPES */